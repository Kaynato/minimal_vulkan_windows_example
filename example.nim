#[ 
  Reproduce an example of a compute shader.
  Messy for now. I'll crush this down to increase reusability and quality...
]#

# import nimgl, nimgl/vulkan

{.experimental: "codeReordering".}


when defined(windows):
  {.link: "C:/Windows/System32/vulkan-1.dll"}
elif defined(macosx):
  # TODO probably check these a bit more carefully
  {.link: "libMoltenVK.dylib"}
else:
  # TODO probably check these a bit more carefully
  {.link: "libvulkan.so.1"}

import lib/vulkannim
import nimpng
import sequtils
import math

const
  WIDTH = 3200
  HEIGHT = 2400
  WORKGROUP_SIZE = 32 # TODO goes inside

const SHADER_CODE = slurp("./shaders/mandelbrot.spv")



# For debug only
template vkCheck(value: VkResult) =
  assert value == vkSuccess, "Fatal : VkResult incorrect."

when defined(NDEBUG):
  const enableValidationLayers = false
else:
  const enableValidationLayers = true



# A hacky workaround. Introduces overhead.
# TODO - remove with overhead-free option
type cVector[T] = ref object
    len: uint32
    data: cstringArray

# This is so bad.
template toCStringVector(s: openarray[string]): cVector[cstring] =
  cVector[cstring](len: s.len.uint32, data: allocCStringArray(s))


# Evil, awful debug callback function
let debugReportCallbackFn = proc(
    flags: VkDebugReportFlagsEXT,
    objectType: VkDebugReportObjectTypeEXT,
    cbObject: uint64,
    location: csize,
    messageCode: int32,
    pLayerPrefix: cstring,
    pMessage: cstring,
    pUserData: pointer): VkBool32 {.cdecl.} = 
  # TODO Something about this doesn't seem quite right.
  echo "Debug Report", $pLayerPrefix, $pMessage
  return vkFalse


# We can do better than this... why stuff things into Pixel when we directly read the buffer?
type
  Pixel* = tuple[r, g, b, a: float32]


type ComputeApplication* = ref object of RootObj
  # Vulkan instance
  instance: VkInstance
  debugReportCallback: VkDebugReportCallbackEXT
  # The device that supports vulkan (usually GPU)
  physicalDevice: VkPhysicalDevice
  # Logical device
  device: VkDevice

  # Define the graphics pipeline.
  # It'll be the compute one here.
  pipeline: VkPipeline
  pipelineLayout: VkPipelineLayout
  computeShaderModule: VkShaderModule

  # Allocate the command buffer for the queue pool.
  commandPool: VkCommandPool
  commandBuffer: VkCommandBuffer

  # Descriptors are shader resources.
  # Uniform buffers, storage buffers, images in GLSL.
  # Each descriptor is a resource and can be organized into sets.
  descriptorPool: VkDescriptorPool
  descriptorSet: VkDescriptorSet
  descriptorSetLayout: VkDescriptorSetLayout

  # We'll render to this buffer and establish its backing memory.
  buffer: VkBuffer
  bufferMemory: VkDeviceMemory
  # Buffer size in bytes
  bufferSize: uint32

  enabledLayers: cVector[cstring]
  enabledExtensions: cVector[cstring] # TODO temporary measure

  # Command queue for execution on the device.
  queue: VkQueue
  queueFamilyIndex: uint32


proc run*(app: var ComputeApplication) =
  app.bufferSize = uint32(sizeof(Pixel) * WIDTH * HEIGHT)

  # Initialize vulkan:
  app.createInstance()
  app.findPhysicalDevice()
  app.createDevice()
  app.createBuffer()
  app.createDescriptorSetLayout()
  app.createDescriptorSet()
  app.createComputePipeline()
  app.createCommandBuffer()

  # Finally, run the recorded command buffer.
  app.runCommandBuffer()

  # The former command rendered a mandelbrot set to a buffer.
  # Save that buffer as a png on disk.
  app.saveRenderedImage()

  # Clean up all vulkan resources.
  app.cleanup()


# Gross but maybe vaguely excusable.
proc strcmp(chararray: openarray[char], str: string): bool =
  for i in 0..<chararray.len:
    if chararray[i] == 0.char:
      return i == str.len
    elif chararray[i] != str[i]:
      return false
  return false

# Ultra-awful. Use when testing only.
proc toNimString(chararray: openarray[char]): string =
  result = newStringOfCap(chararray.len)
  for i in 0..<chararray.len:
    if chararray[i] == 0.char:
      return result[0..<i]
    else:
      result.add(chararray[i])
  return result

proc createInstance(app: var ComputeApplication) =
  var enabledExtensionsSeq: seq[string]
  var enabledLayersSeq: seq[string]

  when enableValidationLayers:

    # We get all supported layers with vkEnumerateInstanceLayerProperties.
    var layerCount: uint32
    discard vkEnumerateInstanceLayerProperties(addr layerCount, nil)

    var layerProperties = newSeq[VkLayerProperties](layerCount)
    discard vkEnumerateInstanceLayerProperties(addr layerCount, addr layerProperties[0])

    # Then simply check if VK_LAYER_LUNARG_standard_validation is in the supported layers.
    # Maybe make a names iterator that gives strings...?
    var foundLayer = layerProperties.anyIt(it.layerName.strcmp("VK_LAYER_LUNARG_standard_validation"))
    assert foundLayer, "Layer VK_LAYER_LUNARG_standard_validation not supported\n"
    # Safe to use layer
    enabledLayersSeq.add("VK_LAYER_LUNARG_standard_validation")

    # We need to enable an extension named VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
    # in order to be able to print the warnings emitted by the validation layer.
    # So again, we just check if the extension is among the supported extensions.
    var extensionCount: uint32
    discard vkEnumerateInstanceExtensionProperties(nil, addr extensionCount, nil)

    var extensionProperties = newSeq[VkExtensionProperties](extensionCount)
    discard vkEnumerateInstanceExtensionProperties(nil, addr extensionCount, addr extensionProperties[0])

    var foundExtension = extensionProperties.anyIt(it.extensionName.strcmp(vkExtDebugReportExtensionName))
    assert foundExtension, "Extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME not supported\n"
    enabledExtensionsSeq.add(vkExtDebugReportExtensionName)

  # Create instance
  # Contains application info. This is actually not that important.
  # The only real important field is apiVersion.
  var applicationInfo = VkApplicationInfo(
    sType: vkStructureTypeApplicationInfo,
    pApplicationName: "TEST_VULKAN_COMPUTE_PROGRAM",
    applicationVersion: 0,
    pEngineName: "TEST_VULKAN_ENGINE",
    engineVersion: 0,
    apiVersion: vkApiVersion1_0)
  
  app.enabledLayers = enabledLayersSeq.toCStringVector()
  app.enabledExtensions = enabledExtensionsSeq.toCStringVector()
  var createInfo = VkInstanceCreateInfo(
    sType: vkStructureTypeInstanceCreateInfo,
    flags: 0,
    pApplicationInfo: applicationInfo.addr,
    # Give our desired layers and extensions to vulkan.
    enabledLayerCount: app.enabledLayers.len,
    ppEnabledLayerNames: app.enabledLayers.data,
    enabledExtensionCount: app.enabledExtensions.len,
    ppEnabledExtensionNames: app.enabledExtensions.data)

  # Finally create the actual instance.
  vkCheck(vkCreateInstance(createInfo.addr, nil, app.instance.addr))

  # Register debug callback
  when enableValidationLayers:
    var createDebugInfo = VkDebugReportCallbackCreateInfoEXT(
      sType: vkStructureTypeDebugReportCallbackCreateInfoExt,
      flags: vkDebugReportErrorBitExt or vkDebugReportWarningBitExt or vkDebugReportPerformanceWarningBitExt,
      pfnCallback: debugReportCallbackFn)

    # VkDebugReportCallbackEXT

    # We have to explicitly load this function.
    var instance = app.instance
    vk(vkCreateDebugReportCallbackEXT, instance)
    assert not vvkCreateDebugReportCallbackEXT.isNil, "Could not load vkCreateDebugReportCallbackEXT"

    # Create and register callback.
    vkCheck(vvkCreateDebugReportCallbackEXT(app.instance, createDebugInfo.addr, nil, app.debugReportCallback.addr))

  # TODO know if they are actually de-allocated at some point, if even by vulkan


proc findPhysicalDevice(app: var ComputeApplication) =
  # Find a physical device usable with Vulkan.
  var deviceCount: uint32
  discard vkEnumeratePhysicalDevices(app.instance, deviceCount.addr, nil)
  assert deviceCount > 0.uint32, "Couldn't find a device that supports Vulkan."

  var devices = newSeq[VkPhysicalDevice](deviceCount)
  discard vkEnumeratePhysicalDevices(app.instance, deviceCount.addr, devices[0].addr)

  # Find device we can use. We'd filter it by feature checks, but... there aren't features here.
  for device in devices:
    if true:  # no checks...
      app.physicalDevice = device
      break


proc getComputeQueueFamilyIndex(physicalDevice: VkPhysicalDevice): uint32 =
  var queueFamilyCount: uint32

  # Retrieve all queue families.
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, queueFamilyCount.addr, nil)
  var queueFamilies = newSeq[VkQueueFamilyProperties](queueFamilyCount)
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, queueFamilyCount.addr, queueFamilies[0].addr)

  # Now find a family that supports compute.
  var found: bool = false
  for i, props in queueFamilies:
    if (props.queueCount > 0'u32) and (props.queueFlags and vkQueueComputeBit) != 0:
        # Found a queue with compute.
        found = true
        result = i.uint32
        break

  assert found, "Couldn't find a queue family that supports operations"

proc createDevice(app: var ComputeApplication) =
  # Create the logical device.

  # We only have one queue, so priority is not that imporant. 
  var queuePriorities: cfloat = 1.0
  # find queue family with compute capability.
  var queueFamilyIndex = getComputeQueueFamilyIndex(app.physicalDevice)
  var queueCreateInfo = VkDeviceQueueCreateInfo(
    sType: vkStructureTypeDeviceQueueCreateInfo,
    queueFamilyIndex: queueFamilyIndex,
    # create one queue in this family. We don't need more.
    queueCount: 1,
    pQueuePriorities: queuePriorities.addr)
  
  # Specify any desired device features here. We do not need any for this application, though.
  var deviceFeatures: VkPhysicalDeviceFeatures

  # Now we create the logical device.
  # The logical device allows us to interact with the physical device.
  var deviceCreateInfo = VkDeviceCreateInfo(
    sType: vkStructureTypeDeviceCreateInfo,
    # need to specify validation layers here also.
    enabledLayerCount: app.enabledLayers.len,
    ppEnabledLayerNames: app.enabledLayers.data,
    # when creating the logical device, we also specify its queues.
    pQueueCreateInfos: queueCreateInfo.addr,
    queueCreateInfoCount: 1,
    pEnabledFeatures: deviceFeatures.addr)

  vkCheck(vkCreateDevice(app.physicalDevice, deviceCreateInfo.addr, nil, app.device.addr)) # create logical device.

  # Get a handle to the only member of the queue family.
  vkGetDeviceQueue(app.device, queueFamilyIndex, 0, app.queue.addr)



# find memory type with desired properties.
proc findMemoryType(physicalDevice: VkPhysicalDevice, memoryTypeBits: uint32, properties: VkMemoryPropertyFlags): uint32 =
  var memoryProperties: VkPhysicalDeviceMemoryProperties
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, memoryProperties.addr)

  # How does this search work?
  # See the documentation of VkPhysicalDeviceMemoryProperties for a detailed description. 
  var memMatches: bool
  var propMatches: bool
  for i in 0..<memoryProperties.memoryTypeCount:
    memMatches = (memoryTypeBits and (1'u32 shl i)) != 0
    if memMatches and ((memoryProperties.memoryTypes[i].propertyFlags and properties) == properties):
      return i
  return high(uint32)

proc createBuffer(app: var ComputeApplication) =
  # We will now create a buffer.
  # We will render the mandelbrot set into this buffer in a computer shade later. 
  var bufferCreateInfo = VkBufferCreateInfo(
    sType: vkStructureTypeBufferCreateInfo,
    # buffer size in bytes. 
    size: app.bufferSize,
    # buffer is used as a storage buffer.
    usage: vkBufferUsageStorageBufferBit,
    # buffer is exclusive to a single queue family at a time. 
    sharingMode: vkSharingModeExclusive)

  vkCheck(vkCreateBuffer(app.device, bufferCreateInfo.addr, nil, app.buffer.addr)) # create buffer.

  echo "Allocated ", $bufferCreateInfo.size, " bytes to buffer."

  # But the buffer doesn't allocate memory for itself, so we must do that manually.
  # First, we find the memory requirements for the buffer.
  var memoryRequirements: VkMemoryRequirements 
  vkGetBufferMemoryRequirements(app.device, app.buffer, memoryRequirements.addr)
  
  # Now use obtained memory requirements info to allocate the buffer memory.
  var allocateInfo = VkMemoryAllocateInfo(
    sType: vkStructureTypeMemoryAllocateInfo,
    # Specify the allocation size.
    allocationSize: memoryRequirements.size,
    # There are several types of memory that can be allocated, and we must choose one that:
    # 1) Satisfies the memory requirements (memoryRequirements.memoryTypeBits). 
    # 2) Satifies our own usage requirements.
    #    We want to be able to read the buffer memory from the GPU to the CPU with vkMapMemory,
    #    so we set vkMemoryPropertyHostVisibleBit. 
    # Also, by setting vkMemoryPropertyHostCoherentBit, memory written by the device(GPU) will be easily 
    # visible to the host(CPU), without having to call any extra flushing commands.
    # So mainly for convenience, we set this flag.
    memoryTypeIndex: findMemoryType(
      app.physicalDevice,
      memoryRequirements.memoryTypeBits,
      vkMemoryPropertyHostCoherentBit or vkMemoryPropertyHostVisibleBit
      )
    )

  vkCheck(vkAllocateMemory(app.device, allocateInfo.addr, nil, app.bufferMemory.addr)) # allocate memory on device.
  
  # Now associate that allocated memory with the buffer. With that, the buffer is backed by actual memory. 
  vkCheck(vkBindBufferMemory(app.device, app.buffer, app.bufferMemory, 0))


proc createDescriptorSetLayout(app: var ComputeApplication) =
  # Here we specify a descriptor set layout.
  # This allows us to bind our descriptors to resources in the shader. 

  # Here we specify a binding of type VK_DESCRIPTOR_TYPE_STORAGE_BUFFER to the binding point 0.
  # This binds to `layout(std140, binding = 0) buffer buf` in the compute shader.
  var descriptorSetLayoutBinding = VkDescriptorSetLayoutBinding(
    # binding = 0, as in layout
    binding: 0,
    descriptorType: vkDescriptorTypeStorageBuffer,
    descriptorCount: 1,
    stageFlags: vkShaderStageComputeBit)

  var descriptorSetLayoutCreateInfo = VkDescriptorSetLayoutCreateInfo(
    sType: vkStructureTypeDescriptorSetLayoutCreateInfo,
    bindingCount: 1,
    pBindings: descriptorSetLayoutBinding.addr)

  # Create the descriptor set layout. 
  vkCheck(vkCreateDescriptorSetLayout(
    app.device, descriptorSetLayoutCreateInfo.addr, nil, app.descriptorSetLayout.addr))


proc createDescriptorSet(app: var ComputeApplication) =
  # So we will allocate a descriptor set here.
  # But we need to first create a descriptor pool to do that. 

  # Our descriptor pool can only allocate a single storage buffer.
  var descriptorPoolSize = VkDescriptorPoolSize(
    `type`: vkDescriptorTypeStorageBuffer,
    descriptorCount: 1)

  var descriptorPoolCreateInfo = VkDescriptorPoolCreateInfo(
    sType: vkStructureTypeDescriptorPoolCreateInfo,
    # we only need to allocate one descriptor set from the pool.
    maxSets: 1, 
    poolSizeCount: 1,
    pPoolSizes: descriptorPoolSize.addr)

  # create descriptor pool.
  vkCheck(vkCreateDescriptorPool(app.device, descriptorPoolCreateInfo.addr, nil, app.descriptorPool.addr))

  # With the pool allocated, we can now allocate the descriptor set. 
  var descriptorSetAllocateInfo = VkDescriptorSetAllocateInfo(
    sType: vkStructureTypeDescriptorSetAllocateInfo, 
    # pool to allocate from.
    descriptorPool: app.descriptorPool, 
    # allocate a single descriptor set.
    descriptorSetCount: 1, 
    pSetLayouts: app.descriptorSetLayout.addr)

  # allocate descriptor set.
  vkCheck(vkAllocateDescriptorSets(app.device, descriptorSetAllocateInfo.addr, app.descriptorSet.addr))

  # Next, we need to connect our actual storage buffer with the descriptor. 
  # We use vkUpdateDescriptorSets() to update the descriptor set.
  # Specify the buffer to bind to the descriptor.
  var descriptorBufferInfo = VkDescriptorBufferInfo(
    buffer: app.buffer,
    offset: 0,
    range: app.bufferSize)

  var writeDescriptorSet = VkWriteDescriptorSet(
    sType: vkStructureTypeWriteDescriptorSet,
    # write to this descriptor set.
    dstSet: app.descriptorSet,
    # write to the first, and only binding.
    dstBinding: 0,
    # update a single descriptor.
    descriptorCount: 1,
    # storage buffer.
    descriptorType: vkDescriptorTypeStorageBuffer,
    pBufferInfo: descriptorBufferInfo.addr)

  # perform the update of the descriptor set.
  vkUpdateDescriptorSets(app.device, 1, writeDescriptorSet.addr, 0, nil)

proc createShaderModule(device: VkDevice): VkShaderModule =
  var ptrShader = cast[seq[uint32]](SHADER_CODE)
  ptrShader.setLen((SHADER_CODE.len/4).ceil.int)
  var createInfo = VkShaderModuleCreateInfo(
    sType: vkStructureTypeShaderModuleCreateInfo,
    codeSize: SHADER_CODE.len,
    pCode: ptrShader[0].addr)
  vkCheck(device.vkCreateShaderModule(addr createInfo, nil, addr result))

proc createComputePipeline(app: var ComputeApplication) =
  app.computeShaderModule = createShaderModule(app.device)

  # Now let us actually create the compute pipeline.
  # A compute pipeline is very simple compared to a graphics pipeline.
  # It only consists of a single stage with a compute shader. 
  # So first we specify the compute shader stage, and it's entry point(main).
  var shaderStageCreateInfo = VkPipelineShaderStageCreateInfo(
    sType: vkStructureTypePipelineShaderStageCreateInfo,
    stage: vkShaderStageComputeBit,
    module: app.computeShaderModule,
    pName: "main")

  # The pipeline layout allows the pipeline to access descriptor sets. 
  # So we just specify the descriptor set layout we created earlier.
  var pipelineLayoutCreateInfo = VkPipelineLayoutCreateInfo(
    sType: vkStructureTypePipelineLayoutCreateInfo,
    setLayoutCount: 1,
    pSetLayouts: app.descriptorSetLayout.addr)

  vkCheck(vkCreatePipelineLayout(app.device, pipelineLayoutCreateInfo.addr, nil, app.pipelineLayout.addr))

  var pipelineCreateInfo = VkComputePipelineCreateInfo(
    sType: vkStructureTypeComputePipelineCreateInfo,
    stage: shaderStageCreateInfo,
    layout: app.pipelineLayout)

  # Now, we finally create the compute pipeline. 
  vkCheck(vkCreateComputePipelines(app.device, vkNullHandle, 1, pipelineCreateInfo.addr, nil, app.pipeline.addr))


proc createCommandBuffer(app: var ComputeApplication) =

  # We are getting closer to the end. In order to send commands to the device(GPU),
  # we must first record commands into a command buffer.
  # To allocate a command buffer, we must first create a command pool. So let us do that.
  var commandPoolCreateInfo = VkCommandPoolCreateInfo(
    sType: vkStructureTypeCommandPoolCreateInfo,
    flags: 0,
    # the queue family of this command pool. All command buffers allocated from this command pool,
    # must be submitted to queues of this family ONLY. 
    queueFamilyIndex: app.queueFamilyIndex)
  vkCheck(vkCreateCommandPool(app.device, commandPoolCreateInfo.addr, nil, app.commandPool.addr))
  
  # Now allocate a command buffer from the command pool. 
  var commandBufferAllocateInfo = VkCommandBufferAllocateInfo(
    sType: vkStructureTypeCommandBufferAllocateInfo,
    commandPool: app.commandPool, # specify the command pool to allocate from. 
    # if the command buffer is primary, it can be directly submitted to queues. 
    # A secondary buffer has to be called from some primary command buffer, and cannot be directly 
    # submitted to a queue. To keep things simple, we use a primary command buffer. 
    level: vkCommandBufferLevelPrimary,
    # allocate a single command buffer. 
    commandBufferCount: 1)
  vkCheck(vkAllocateCommandBuffers(app.device, commandBufferAllocateInfo.addr, app.commandBuffer.addr)) # allocate command buffer.

  # Now we shall start recording commands into the newly allocated command buffer. 
  var beginInfo = VkCommandBufferBeginInfo(
    sType: vkStructureTypeCommandBufferBeginInfo,
    # the buffer is only submitted and used once in this application.
    flags: vkCommandBufferUsageOneTimeSubmitBit)
  vkCheck(vkBeginCommandBuffer(app.commandBuffer, beginInfo.addr)) # start recording commands.

  # We need to bind a pipeline, AND a descriptor set before we dispatch.
  # The validation layer will NOT give warnings if you forget these, so be very careful not to forget them.
  vkCmdBindPipeline(app.commandBuffer, vkPipelineBindPointCompute, app.pipeline)
  vkCmdBindDescriptorSets(
    app.commandBuffer, vkPipelineBindPointCompute,
    app.pipelineLayout, 0, 1,
    app.descriptorSet.addr, 0, nil)

  
  # Calling vkCmdDispatch basically starts the compute pipeline, and executes the compute shader.
  # The number of workgroups is specified in the arguments.
  # If you are already familiar with compute shaders from OpenGL, this should be nothing new to you.
  let shaderWidth: uint32 = uint32(ceil(WIDTH.float / WORKGROUP_SIZE.float))
  let shaderHeight: uint32 = uint32(ceil(HEIGHT.float / WORKGROUP_SIZE.float))
  vkCmdDispatch(app.commandBuffer, shaderWidth, shaderHeight, 1)
  vkCheck(vkEndCommandBuffer(app.commandBuffer)) # end recording commands.


proc runCommandBuffer(app: var ComputeApplication) =
  # Now we shall finally submit the recorded command buffer to a queue.
  var submitInfo = VkSubmitInfo(
    sType: vkStructureTypeSubmitInfo,
    # submit a single command buffer
    commandBufferCount: 1,
    # the command buffer to submit.
    pCommandBuffers: app.commandBuffer.addr)

  # We create a fence.
  var fence: VkFence
  var fenceCreateInfo = VkFenceCreateInfo(
    sType: vkStructureTypeFenceCreateInfo,
    flags: 0)
  vkCheck(vkCreateFence(app.device, fenceCreateInfo.addr, nil, fence.addr))

  # We submit the command buffer on the queue, at the same time giving a fence.
  vkCheck(vkQueueSubmit(app.queue, 1, submitInfo.addr, fence))

  # The command will not have finished executing until the fence is signalled.
  # So we wait here.
  # We will, directly after this, read our buffer from the GPU,
  # and we will not be sure that the command has finished executing unless we wait for the fence.
  # Hence, we use a fence here.
  vkCheck(vkWaitForFences(app.device, 1, fence.addr, vkTrue, 100000000000'u64))

  vkDestroyFence(app.device, fence, nil)


proc cleanup(app: var ComputeApplication) =
  # Clean up all Vulkan Resources. 

  when enableValidationLayers:
    # Destroy callback.
    let instance = app.instance
    vk(vkDestroyDebugReportCallbackEXT, instance)
    assert not vvkDestroyDebugReportCallbackEXT.isNil, "Could not load vkDestroyDebugReportCallbackEXT"
    vvkDestroyDebugReportCallbackEXT(instance, app.debugReportCallback, nil)

  vkFreeMemory(app.device, app.bufferMemory, nil)
  vkDestroyBuffer(app.device, app.buffer, nil)  
  vkDestroyShaderModule(app.device, app.computeShaderModule, nil)
  vkDestroyDescriptorPool(app.device, app.descriptorPool, nil)
  vkDestroyDescriptorSetLayout(app.device, app.descriptorSetLayout, nil)
  vkDestroyPipelineLayout(app.device, app.pipelineLayout, nil)
  vkDestroyPipeline(app.device, app.pipeline, nil)
  vkDestroyCommandPool(app.device, app.commandPool, nil)  
  vkDestroyDevice(app.device, nil)
  vkDestroyInstance(app.instance, nil)    

  # TODO There must be a better way.
  deallocCStringArray(app.enabledLayers.data)
  deallocCStringArray(app.enabledExtensions.data)

  
proc saveRenderedImage(app: var ComputeApplication) =
  var mappedMemory: pointer
  # Map the buffer memory, from GPU back to CPU.
  discard vkMapMemory(app.device, app.bufferMemory, 0, app.bufferSize, 0, mappedMemory.addr)
  # Painful realities of C interop
  var pmappedMemory = cast[ptr UncheckedArray[Pixel]](mappedMemory)

  # Get the color data from the buffer, and cast it to bytes.
  # We save the data to a vector.
  # Oh come on! String image? Really?
  # var image = newSeq[byte](WIDTH * HEIGHT * 4)
  var image = newStringOfCap(WIDTH * HEIGHT * 4)
  for i in 0 ..< (WIDTH*HEIGHT):
    var pix: Pixel = pmappedMemory[i]
    image.add(char(255.0f * pix.r))
    image.add(char(255.0f * pix.g))
    image.add(char(255.0f * pix.b))
    image.add(char(255.0f * pix.a))

  # Done reading, so unmap.
  vkUnmapMemory(app.device, app.bufferMemory)

  # Now we save the acquired color data to a .png.

  # Why does this use string? Why does this PNG library use string? Are they irresponsible?
  echo "Saving image to file."
  discard savePNG32("mandelbrot.png", image, WIDTH, HEIGHT)



var app = new(ComputeApplication)
app.run()

echo "End of program."
