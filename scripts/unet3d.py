from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

"""
def Conv3x3Stack(in_channels, out_channels, negative_slope):
	outputs = nn.Sequential(nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=0, dilation=1, groups=1, bias=True),
	                        nn.LeakyReLU(negative_slope),
	                        nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=0, dilation=1, groups=1, bias=True),
	                        nn.LeakyReLU(negative_slope))
	init.kaiming_normal_(outputs[0].weight, negative_slope, 'fan_in', 'leaky_relu'); init.constant_(outputs[0].bias, 0.0)
	init.kaiming_normal_(outputs[2].weight, negative_slope, 'fan_in', 'leaky_relu'); init.constant_(outputs[2].bias, 0.0)
	return outputs


def DConv3x3Stack(in_channels, out_channels, negative_slope):
	outputs = nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=1, bias=True),
	                        nn.Conv3d(out_channels, out_channels, 1, stride=1, padding=0, dilation=1, groups=1, bias=True),
	                        nn.LeakyReLU(negative_slope))
	init.kaiming_normal_(outputs[0].weight, 0, 'fan_in', 'relu'); init.constant_(outputs[0].bias, 0.0)
	init.kaiming_normal_(outputs[1].weight, negative_slope, 'fan_in', 'leaky_relu'); init.constant_(outputs[1].bias, 0.0)
	return outputs
"""

class Conv3x3Stack(nn.Module):
	def __init__(self, in_channels, out_channels, negative_slope):
		super(Conv3x3Stack, self).__init__()
		self.block = nn.Sequential(nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=0, dilation=1, groups=1, bias=True),
		                           nn.LeakyReLU(negative_slope),
		                           nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=0, dilation=1, groups=1, bias=True),
		                           nn.LeakyReLU(negative_slope))
		init.kaiming_normal_(self.block[0].weight, negative_slope, 'fan_in', 'leaky_relu'); init.constant_(self.block[0].bias, 0.0)
		init.kaiming_normal_(self.block[2].weight, negative_slope, 'fan_in', 'leaky_relu'); init.constant_(self.block[2].bias, 0.0)
	
	def forward(self, input):
		return self.block(input)


class DConv3x3Stack(nn.Module):
	def __init__(self, in_channels, out_channels, negative_slope):
		super(DConv3x3Stack, self).__init__()
		self.block = nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=1, bias=True),
		                           nn.Conv3d(out_channels, out_channels, 1, stride=1, padding=0, dilation=1, groups=1, bias=True),
		                           nn.LeakyReLU(negative_slope))
		init.kaiming_normal_(self.block[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.block[0].bias, 0.0)
		init.kaiming_normal_(self.block[1].weight, negative_slope, 'fan_in', 'leaky_relu'); init.constant_(self.block[1].bias, 0.0)
	
	def forward(self, input):
		return self.block(input)


class UNet3D(nn.Module):
	def __init__(self, out_channel=3, negative_slope=0.005, type='normal'):
		super(UNet3D, self).__init__()
		self.pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))
		if type == 'debug':
			base_channel = 8
			grow_rate = 2
		elif type == 'nano':
			base_channel = 16
			grow_rate = 4
		elif type == 'normal':
			base_channel = 12
			grow_rate = 5
		else: raise NotImplementedError
		self.convs1 = Conv3x3Stack(1, base_channel, negative_slope)
		self.convs2 = Conv3x3Stack(base_channel, base_channel * grow_rate, negative_slope)
		self.convs3 = Conv3x3Stack(base_channel * grow_rate, base_channel * grow_rate ** 2, negative_slope)
		self.convs4 = Conv3x3Stack(base_channel * grow_rate ** 2, base_channel * grow_rate ** 3, negative_slope)
		
		self.dconvs1 = DConv3x3Stack(base_channel * grow_rate ** 3, base_channel * grow_rate ** 2, negative_slope)
		self.dconvs2 = DConv3x3Stack(base_channel * grow_rate ** 2, base_channel * grow_rate, negative_slope)
		self.dconvs3 = DConv3x3Stack(base_channel * grow_rate, base_channel, negative_slope)
		self.convs5 = Conv3x3Stack(base_channel * grow_rate ** 2 * 2, base_channel * grow_rate ** 2, negative_slope)
		self.convs6 = Conv3x3Stack(base_channel * grow_rate * 2, base_channel * grow_rate, negative_slope)
		self.convs7 = Conv3x3Stack(base_channel * 2, base_channel, negative_slope)
		
		self.conv8 = nn.Conv3d(base_channel, out_channel, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		# self.dconvs1_2 = DConv3x3Stack(base_channel * grow_rate ** 3, base_channel * grow_rate ** 2, negative_slope)
		# self.dconvs2_2 = DConv3x3Stack(base_channel * grow_rate ** 2, base_channel * grow_rate, negative_slope)
		# self.dconvs3_2 = DConv3x3Stack(base_channel * grow_rate, base_channel, negative_slope)
		# self.convs5_2 = Conv3x3Stack(base_channel * grow_rate ** 2 * 2, base_channel * grow_rate ** 2, negative_slope)
		# self.convs6_2 = Conv3x3Stack(base_channel * grow_rate * 2, base_channel * grow_rate, negative_slope)
		# self.convs7_2 = Conv3x3Stack(base_channel * 2, 1, negative_slope)
	
	def crop_and_concat(self, upsampled, bypass, crop=False):
		if crop:
			c = (bypass.size()[3] - upsampled.size()[3]) // 2
			cc = (bypass.size()[2] - upsampled.size()[2]) // 2
			assert(c > 0)
			assert(cc > 0)
			bypass_ = F.pad(bypass, (-c, -c, -c, -c, -cc, -cc))
		return torch.cat((upsampled, bypass_), 1)
	
	def forward(self, inputs):
		convs1 = self.convs1(inputs)
		feat = self.pool(convs1)
		convs2 = self.convs2(feat)
		feat = self.pool(convs2)
		convs3 = self.convs3(feat)
		feat = self.pool(convs3)
		convs4 = self.convs4(feat)

		mc1 = self.crop_and_concat(self.dconvs1(convs4), convs3, crop=True)
		convs5 = self.convs5(mc1)
		mc2 = self.crop_and_concat(self.dconvs2(convs5), convs2, crop=True)
		convs6 = self.convs6(mc2)
		mc3 = self.crop_and_concat(self.dconvs3(convs6), convs1, crop=True)
		convs7 = self.convs7(mc3)

		conv8 = self.conv8(convs7)
		output = F.sigmoid(conv8)
		return output


if __name__ == '__main__':
	""" example of weight sharing """
	#self.convs1_siamese = Conv3x3Stack(1, 12, negative_slope)
	#self.convs1_siamese[0].weight = self.convs1[0].weight
	
	import numpy as np
	model = UNet3D().to('cuda:0')
	# torch.save(model, 'model3.ckpt')
	
	x = torch.tensor(np.random.random((1, 1, 84, 268, 268)).astype(np.float32)).to('cuda:0')
	out = model(x)
	print(out.shape) # (1, 3, 56, 56, 56)
