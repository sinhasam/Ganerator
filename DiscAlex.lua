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
	numEpoch = 10000,
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



local function weightsInit(m)
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
--features:add(nn.ReLU(true)) -- HERERERERERE


features:add(spatialConv(64,192,5,5,1,1,2,2))	--27 -> 27
features:add(spatialMaxPool(3,3,2,2))           --27 -> 13
features:add(cudnn.ReLU(true))
--features:add(nn.ReLU(true)) -- HERERERERERE
features:add(spatialConv(192,384,3,3,1,1,1,1))  --13 -> 13
features:add(cudnn.ReLU(true))
--features:add(nn.ReLU(true)) --HERERERERERERE

features:add(spatialConv(384,256,3,3,1,1,1,1))  --13 -> 13
features:add(cudnn.ReLU(true))
--features:add(nn.ReLU(true)) --HERERERERER
features:add(spatialConv(256,256,3,3,1,1,1,1))  --13 -> 13
features:add(spatialMaxPool(3,3,2,2))           --13 -> 6
features:add(cudnn.ReLU(true))
--features:add(nn.ReLU(true)) -- HEREREREERE
--features:add(batchNorm(256,nil,nil,false))

print('Training!')
print('Setting AlexNet')

classifier:add(nn.View(256*6*6))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(256*6*6, 4096))
classifier:add(nn.Threshold(0, 1e-6))
--classifier:add(batchNorm(4096,nil,nil,false))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(4096, 4096))
classifier:add(nn.Threshold(0, 1e-6))
--classifier:add(batchNorm(4096,nil,nil,false))
classifier:add(nn.Linear(4096, 1))
classifier:add(nn.LogSoftMax())

local DiscriminativeModel = nn.Sequential()
print('Setting up the discriminative model')

DiscriminativeModel:add(features):add(classifier)

DiscriminativeModel:apply(weightsInit)


local GenerativeModel = nn.Sequential()

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


GenerativeModel:add(nn.Tanh())

GenerativeModel:apply(weightsInit)
local criterion = nn.BCECriterion()

optimStateG = {
	lr = opt.lr,
	b1 = opt.beta1,
}
optimStateD = {
	lr = opt.lr,
	b1 = opt.beta1,
}

local input = torch.Tensor(opt.batchSize, 3, 224, 224)
local noise = torch.rand(opt.batchSize, 100, 1, 1)
local label = torch.Tensor(opt.batchSize)
local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()


if opt.gpu > 0 then
	cutorch.setDevice(opt.gpu)
	input = input:cuda()
	noise = noise:cuda()
	label = label:cuda()
	
	if pcall(require,'cudnn') then
		require 'cudnn'
		cudnn.benchmark = true
		cudnn.convert(GenerativeModel, cudnn)
		cudnn.convert(DiscriminativeModel, cudnn)
	end
	DiscriminativeModel:cuda()
	GenerativeModel:cuda()
	criterion:cuda()
end

local dParams, dGradParams = DiscriminativeModel:getParameters()
local gParams, gGradParams = GenerativeModel:getParameters()

noiseNorm = noise:clone()
-- noiseNorm:normal(0,1)

imgCount = 1
dataset = {}
for img in lfs.dir("images") do
	i, j = string.find(img,"JPEG")
	if(i ~= nil and j~=nil) then
		dataset[imgCount] = img
		imgCount = imgCount + 1
	end
end
dataSize = imgCount
imgCount = 1


local fDx = function(x)
	dGradParams:zero()

	data_tm:reset()
	data_tm:resume()
	local img = image.load(("images/"..dataset[imgCount]),3,float)
	data_tm:stop()
	input:copy(img)
	label:fill(1)


	local output = DiscriminativeModel:forward(input)
	local imgError = criterion:forward(output, label)
	local dError = criterion:backward(output, label)

	DiscriminativeModel:backward(input, dError)

	noise:normal(0,1)
	genImg = GenerativeModel:forward(noise)
	input:copy(genImg)
	label:fill(0)
	
    local output = DiscriminativeModel:forward(input)
    local genImgError = criterion:forward(output, label)
    local dGenError = criterion:backward(output, label)
	DiscriminativeModel:backward(input, dGenError)

	dError = imgError + genImgError
	
	return dError, dGradParams
end

local fGx = function(x)
	gGradParams:zero()
	label:fill(1)

	genOutput = DiscriminativeModel.output
	gError = criterion:forward(genOutput, label)
	local dError = criterion:backward(genOutput, label)
	local dGenError = DiscriminativeModel:updateGradInput(input, dError)
	
	GenerativeModel:backward(noise,dGenError)
	return gError, gGradParams
end

for epoch = 1, opt.numEpoch do
	epoch_tm:reset()
	local counter = 0
	for i = 1, dataSize, opt.batchSize do
		tm:reset()
		print("loop")
		optim.adam(fDx, dParams, optimStateD)
		optim.adam(fGx, gParams, optimStateG)
		print("passedloop")
		counter = counter + 1
		print(counter)
		if counter % 50 == 0 then
			print("Image:", counter)
		end
	end
	print("Epoch Time:", epoch_tm:time().real)
	print(genImg:nDimension())
	print(genImg:size())
	gGradParams, dGradParams, gParams, dParams = nil, nil, nil, nil
	torch.save("TrainedModels2/Gen", GenerativeModel:clearState())
	torch.save("TrainedModels2/Disc", DiscriminativeModel:clearState())
--	local img = image.toDisplayTensor(genImg)
	--print(#genImg[1])
--	print(genImg)
--	image.save("TrainedModels/" ..epoch.. "genImg.jpeg", genImg[1])
	gParams, gGradParams = GenerativeModel:getParameters()
	dParams, dGradParams = DiscriminativeModel:getParameters()

	image.save("TrainedModels2/" ..epoch.. "genImg.jpeg", input[1])
end

