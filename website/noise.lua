require 'cudnn'
require 'cunn'
require 'nn'
require 'paths'
require 'optim'
require 'torch'
require 'lfs'
require 'optim'
require 'image'


opt = {
        batchSize = 1,
        lr = 0.02,
        b1 = 0.5,
        numEpoch = 50,
        gpu = 1
}

cutorch.setHeapTracking(true)
local spatialFullConvolution = nn.SpatialFullConvolution
local spatialConv = cudnn.SpatialConvolution
local spatialMaxPool = cudnn.SpatialMaxPooling
--local spatialConv = nn.SpatialConvolution --HERERERERERE
--local spatialMaxPool = nn.SpatialMaxPooling --HERERERERERERE
local features  = nn.Sequential()
local classifier = nn.Sequential()
local batchNorm = nn.SpatialBatchNormalization
torch.setdefaulttensortype('torch.FloatTensor')
local GenerativeModel = nn.Sequential()
--[[
GenerativeModel:add(spatialFullConvolution(100, 64 * 32, 4, 4))
GenerativeModel:add(batchNorm(64 * 32)):add(nn.ReLU(true))
-- state size: (64 * 32) x 4 x 4
GenerativeModel:add(spatialFullConvolution(64 * 32, 64 * 16, 4, 4, 2, 2, 1, 1))
GenerativeModel:add(batchNorm(64 * 16)):add(nn.ReLU(true))
-- state size: (64 * 16) x 8 x 8
GenerativeModel:add(spatialFullConvolution(64 * 16, 64 * 8, 4, 4, 2, 2, 1, 1))
GenerativeModel:add(batchNorm(64 * 8)):add(nn.ReLU(true))
-- state size: (64 * 8) x 16 x 16
GenerativeModel:add(spatialFullConvolution(64 * 8, 64 * 4, 3, 3, 2, 2, 2, 2))
GenerativeModel:add(batchNorm(64 * 4)):add(nn.ReLU(true))
-- state size: (64 * 4) x 29 x 29
GenerativeModel:add(spatialFullConvolution(64 * 4, 64 * 2, 4, 4, 2, 2, 2, 2))
GenerativeModel:add(batchNorm(64 * 2)):add(nn.ReLU(true))
-- state size: (64 * 2) x 56 x 56
GenerativeModel:add(spatialFullConvolution(64 * 2, 64, 4, 4, 2, 2, 1, 1))
GenerativeModel:add(batchNorm(64)):add(nn.ReLU(true))
-- state size: (64) x 112 x 112
GenerativeModel:add(spatialFullConvolution(64, 3, 4, 4, 2, 2, 1, 1))
-- state size: 3 x 224 x 224
]]--

GenerativeModel:add(nn.Tanh())

gen = torch.load("Gen")
weights, grads = gen:getParameters()
GenerativeModel:apply(weights)
local noise = torch.rand(opt.batchSize, 100, 1, 1)

genImg = weights:forward(noise)


image.save("genImage.jpeg", input[1])


