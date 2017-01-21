require 'cudnn'
require 'cunn'
require 'nn'

local SpatialFullConvolution = nn.SpatialFullConvolution
local spatialConv = cudnn.SpatialConvolution
local spatialMaxPool = cudnn.SpatialMaxPooling
local features  = nn.Sequential()
local classifier = nn.Sequential()
local batchNorm = nn.SpatialBatchNormalization



local function weights_init(m)
	local name = torch.type(m)
	if name:find('Conv') then
		m.weight:normal(0.0,0.02)
	elseif name:find('batchNorm') then
		m.weight:normal(1.0,0.02)
	end
end



features:add(spatialConv(3,64,11,11,4,4,2,2))	--224 -> 55
features:add(spatialMaxPool(3,3,2,2))
features:add(cudnn.ReLU(true))

features:add(spatialConv(64,192,5,5,1,1,2,2))	--27 -> 27
features:add(spatialMaxPool(3,3,2,2))           --27 -> 13
features:add(cudnn.ReLU(true))

features:add(spatialConv(192,384,3,3,1,1,1,1))  --13 -> 13
features:add(cudnn.ReLU(true))

features:add(spatialConv(384,256,3,3,1,1,1,1))  --13 -> 13
features:add(cudnn.ReLU(true))

features:add(spatialConv(256,256,3,3,1,1,1,1))  --13 -> 13
features:add(spatialMaxPool(3,3,2,2))           --13 -> 6
features:add(cudnn.ReLU(true))
features:add(batchNorm(256,nil,nil,false))


classifier:add(nn.View(256*6*6))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(256*6*6, 4096))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(batchNorm(4096,nil,nil,false))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(4096, 4096))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(batchNorm(4096,nil,nil,false))
classifier:add(nn.Linear(4096,1000))
classifier:add(nn.LogSoftMax())

local disc_model = nn.Sequential()

disc_model:add(features):add(classifier)

disc_model:apply(weights_init)


local gen_model = nn.Sequential()

netG:add(SpatialFullConvolution(100, 64 * 32, 4, 4))
netG:add(batchNorm(64 * 32)):add(nn.ReLU(true))
-- state size: (64 * 32) x 4 x 4
netG:add(SpatialFullConvolution(64 * 32, 64 * 16, 4, 4, 2, 2, 1, 1))
netG:add(batchNorm(64 * 16)):add(nn.ReLU(true))
-- state size: (64 * 16) x 8 x 8
netG:add(SpatialFullConvolution(64 * 16, 64 * 8, 4, 4, 2, 2, 1, 1))
netG:add(batchNorm(64 * 8)):add(nn.ReLU(true))
-- state size: (64 * 8) x 16 x 16
netG:add(SpatialFullConvolution(64 * 8, 64 * 4, 3, 3, 2, 2, 2, 2))
netG:add(batchNorm(64 * 4)):add(nn.ReLU(true))
-- state size: (64 * 4) x 29 x 29
netG:add(SpatialFullConvolution(64 * 4, 64 * 2, 4, 4, 2, 2, 2, 2))
netG:add(batchNorm(64 * 2)):add(nn.ReLU(true))
-- state size: (64 * 2) x 56 x 56
netG:add(SpatialFullConvolution(64 * 2, 64, 4, 4, 2, 2, 1, 1))
netG:add(batchNorm(64)):add(nn.ReLU(true))
-- state size: (64) x 112 x 112
netG:add(SpatialFullConvolution(64, 3, 4, 4, 2, 2, 1, 1))
-- state size: 3 x 224 x 224


netG:add(nn.Tanh())

netG:apply(weights_init)


