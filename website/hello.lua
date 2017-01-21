local colors = { 'red', 'blue', 'green', 'orange', 'yellow', 'purple' }
local balls = {}


for _, color in pairs(colors) do
	local radius = math.random(5, 25)
	local circle = createCircle(color, radius)

	local velX = math.random(-8, 8)
	local velY = math.random(-8, 8)

	circle.onUpdate = function ()
		circle.x = circle.x + velX
		circle.y = circle.y + velY

		if circle.x < radius or circle.x > 300 - radius then velX = -velX end
		if circle.y < radius or circle.y > 300 - radius then velY = -velY end
	end
end
