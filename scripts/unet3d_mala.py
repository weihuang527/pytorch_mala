from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class UNet3D_MALA(nn.Module):
	def __init__(self):
		super(UNet3D_MALA, self).__init__()
		self.mala = np.load('./models/net_iter_200000.npy', encoding='bytes').item(0)
		
		self.conv1 = nn.Conv3d(1, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		assert(self.conv1.weight.data.numpy().shape == self.mala[b'conv1_w'].shape)
		assert(self.conv1.bias.data.numpy().shape == self.mala[b'conv1_b'].shape)
		self.conv1.weight.data = torch.Tensor(self.mala[b'conv1_w'])
		self.conv1.bias.data = torch.Tensor(self.mala[b'conv1_b'])
		
		self.conv2 = nn.Conv3d(12, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		assert(self.conv2.weight.data.numpy().shape == self.mala[b'conv2_w'].shape)
		assert(self.conv2.bias.data.numpy().shape == self.mala[b'conv2_b'].shape)
		self.conv2.weight.data = torch.Tensor(self.mala[b'conv2_w'])
		self.conv2.bias.data = torch.Tensor(self.mala[b'conv2_b'])
		
		self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))
		
		self.conv3 = nn.Conv3d(12, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		assert(self.conv3.weight.data.numpy().shape == self.mala[b'conv3_w'].shape)
		assert(self.conv3.bias.data.numpy().shape == self.mala[b'conv3_b'].shape)
		self.conv3.weight.data = torch.Tensor(self.mala[b'conv3_w'])
		self.conv3.bias.data = torch.Tensor(self.mala[b'conv3_b'])
		
		self.conv4 = nn.Conv3d(60, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		assert(self.conv4.weight.data.numpy().shape == self.mala[b'conv4_w'].shape)
		assert(self.conv4.bias.data.numpy().shape == self.mala[b'conv4_b'].shape)
		self.conv4.weight.data = torch.Tensor(self.mala[b'conv4_w'])
		self.conv4.bias.data = torch.Tensor(self.mala[b'conv4_b'])
		
		self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))
		
		self.conv5 = nn.Conv3d(60, 300, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		assert(self.conv5.weight.data.numpy().shape == self.mala[b'conv5_w'].shape)
		assert(self.conv5.bias.data.numpy().shape == self.mala[b'conv5_b'].shape)
		self.conv5.weight.data = torch.Tensor(self.mala[b'conv5_w'])
		self.conv5.bias.data = torch.Tensor(self.mala[b'conv5_b'])
		
		self.conv6 = nn.Conv3d(300, 300, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		assert(self.conv6.weight.data.numpy().shape == self.mala[b'conv6_w'].shape)
		assert(self.conv6.bias.data.numpy().shape == self.mala[b'conv6_b'].shape)
		self.conv6.weight.data = torch.Tensor(self.mala[b'conv6_w'])
		self.conv6.bias.data = torch.Tensor(self.mala[b'conv6_b'])
		
		self.pool3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))
		
		self.conv7 = nn.Conv3d(300, 1500, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		assert(self.conv7.weight.data.numpy().shape == self.mala[b'conv7_w'].shape)
		assert(self.conv7.bias.data.numpy().shape == self.mala[b'conv7_b'].shape)
		self.conv7.weight.data = torch.Tensor(self.mala[b'conv7_w'])
		self.conv7.bias.data = torch.Tensor(self.mala[b'conv7_b'])
		
		self.conv8 = nn.Conv3d(1500, 1500, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		assert(self.conv8.weight.data.numpy().shape == self.mala[b'conv8_w'].shape)
		assert(self.conv8.bias.data.numpy().shape == self.mala[b'conv8_b'].shape)
		self.conv8.weight.data = torch.Tensor(self.mala[b'conv8_w'])
		self.conv8.bias.data = torch.Tensor(self.mala[b'conv8_b'])
		
		self.dconv1 = nn.ConvTranspose3d(1500, 1500, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=1500, bias=False)
		assert(self.dconv1.weight.data.numpy().shape == self.mala[b'dconv1_w'].shape)
		self.dconv1.weight.data = torch.Tensor(self.mala[b'dconv1_w'])
		
		self.conv9 = nn.Conv3d(1500, 300, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		assert(self.conv9.weight.data.numpy().shape == self.mala[b'conv9_w'].shape)
		assert(self.conv9.bias.data.numpy().shape == self.mala[b'conv9_b'].shape)
		self.conv9.weight.data = torch.Tensor(self.mala[b'conv9_w'])
		self.conv9.bias.data = torch.Tensor(self.mala[b'conv9_b'])
		
		self.conv10 = nn.Conv3d(600, 300, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		assert(self.conv10.weight.data.numpy().shape == self.mala[b'conv10_w'].shape)
		assert(self.conv10.bias.data.numpy().shape == self.mala[b'conv10_b'].shape)
		self.conv10.weight.data = torch.Tensor(self.mala[b'conv10_w'])
		self.conv10.bias.data = torch.Tensor(self.mala[b'conv10_b'])
		
		self.conv11 = nn.Conv3d(300, 300, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		assert(self.conv11.weight.data.numpy().shape == self.mala[b'conv11_w'].shape)
		assert(self.conv11.bias.data.numpy().shape == self.mala[b'conv11_b'].shape)
		self.conv11.weight.data = torch.Tensor(self.mala[b'conv11_w'])
		self.conv11.bias.data = torch.Tensor(self.mala[b'conv11_b'])
		
		self.dconv2 = nn.ConvTranspose3d(300, 300, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=300, bias=False)
		assert(self.dconv2.weight.data.numpy().shape == self.mala[b'dconv2_w'].shape)
		self.dconv2.weight.data = torch.Tensor(self.mala[b'dconv2_w'])
		
		self.conv12 = nn.Conv3d(300, 60, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		assert(self.conv12.weight.data.numpy().shape == self.mala[b'conv12_w'].shape)
		assert(self.conv12.bias.data.numpy().shape == self.mala[b'conv12_b'].shape)
		self.conv12.weight.data = torch.Tensor(self.mala[b'conv12_w'])
		self.conv12.bias.data = torch.Tensor(self.mala[b'conv12_b'])
		
		self.conv13 = nn.Conv3d(120, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		assert(self.conv13.weight.data.numpy().shape == self.mala[b'conv13_w'].shape)
		assert(self.conv13.bias.data.numpy().shape == self.mala[b'conv13_b'].shape)
		self.conv13.weight.data = torch.Tensor(self.mala[b'conv13_w'])
		self.conv13.bias.data = torch.Tensor(self.mala[b'conv13_b'])
		
		self.conv14 = nn.Conv3d(60, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		assert(self.conv14.weight.data.numpy().shape == self.mala[b'conv14_w'].shape)
		assert(self.conv14.bias.data.numpy().shape == self.mala[b'conv14_b'].shape)
		self.conv14.weight.data = torch.Tensor(self.mala[b'conv14_w'])
		self.conv14.bias.data = torch.Tensor(self.mala[b'conv14_b'])
		
		self.dconv3 = nn.ConvTranspose3d(60, 60, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=60, bias=False)
		assert(self.dconv3.weight.data.numpy().shape == self.mala[b'dconv3_w'].shape)
		self.dconv3.weight.data = torch.Tensor(self.mala[b'dconv3_w'])
		
		self.conv15 = nn.Conv3d(60, 12, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		assert(self.conv15.weight.data.numpy().shape == self.mala[b'conv15_w'].shape)
		assert(self.conv15.bias.data.numpy().shape == self.mala[b'conv15_b'].shape)
		self.conv15.weight.data = torch.Tensor(self.mala[b'conv15_w'])
		self.conv15.bias.data = torch.Tensor(self.mala[b'conv15_b'])
		
		self.conv16 = nn.Conv3d(24, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		assert(self.conv16.weight.data.numpy().shape == self.mala[b'conv16_w'].shape)
		assert(self.conv16.bias.data.numpy().shape == self.mala[b'conv16_b'].shape)
		self.conv16.weight.data = torch.Tensor(self.mala[b'conv16_w'])
		self.conv16.bias.data = torch.Tensor(self.mala[b'conv16_b'])
		
		self.conv17 = nn.Conv3d(12, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		assert(self.conv17.weight.data.numpy().shape == self.mala[b'conv17_w'].shape)
		assert(self.conv17.bias.data.numpy().shape == self.mala[b'conv17_b'].shape)
		self.conv17.weight.data = torch.Tensor(self.mala[b'conv17_w'])
		self.conv17.bias.data = torch.Tensor(self.mala[b'conv17_b'])
		
		self.conv18 = nn.Conv3d(12, 3, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		assert(self.conv18.weight.data.numpy().shape == self.mala[b'conv18_w'].shape)
		assert(self.conv18.bias.data.numpy().shape == self.mala[b'conv18_b'].shape)
		self.conv18.weight.data = torch.Tensor(self.mala[b'conv18_w'])
		self.conv18.bias.data = torch.Tensor(self.mala[b'conv18_b'])
	
	def crop_and_concat(self, upsampled, bypass, crop=False):
		if crop:
			c = (bypass.size()[3] - upsampled.size()[3]) // 2
			cc = (bypass.size()[2] - upsampled.size()[2]) // 2
			assert(c > 0)
			assert(cc > 0)
			bypass = F.pad(bypass, (-c, -c, -c, -c, -cc, -cc))
		return torch.cat((upsampled, bypass), 1)
	
	def forward(self, input):
		conv1 = F.leaky_relu(self.conv1(input), 0.005)
		conv2 = F.leaky_relu(self.conv2(conv1), 0.005)
		pool1 = self.pool1(conv2)
		conv3 = F.leaky_relu(self.conv3(pool1), 0.005)
		conv4 = F.leaky_relu(self.conv4(conv3), 0.005)
		pool2 = self.pool2(conv4)
		conv5 = F.leaky_relu(self.conv5(pool2), 0.005)
		conv6 = F.leaky_relu(self.conv6(conv5), 0.005)
		pool3 = self.pool3(conv6)
		conv7 = F.leaky_relu(self.conv7(pool3), 0.005)
		conv8 = F.leaky_relu(self.conv8(conv7), 0.005)
		dconv1 = self.dconv1(conv8)
		conv9 = self.conv9(dconv1)
		mc1 = self.crop_and_concat(conv9, conv6, crop=True)
		conv10 = F.leaky_relu(self.conv10(mc1), 0.005)
		conv11 = F.leaky_relu(self.conv11(conv10), 0.005)
		dconv2 = self.dconv2(conv11)
		conv12 = self.conv12(dconv2)
		mc2 = self.crop_and_concat(conv12, conv4, crop=True)
		conv13 = F.leaky_relu(self.conv13(mc2), 0.005)
		conv14 = F.leaky_relu(self.conv14(conv13), 0.005)
		dconv3 = self.dconv3(conv14)
		conv15 = self.conv15(dconv3)
		mc3 = self.crop_and_concat(conv15, conv2, crop=True)
		conv16 = F.leaky_relu(self.conv16(mc3), 0.005)
		conv17 = F.leaky_relu(self.conv17(conv16), 0.005)
		conv18 = self.conv18(conv17)
		output = F.sigmoid(conv18)
		return output


if __name__ == '__main__':
	""" example of weight sharing """
	#self.convs1_siamese = Conv3x3Stack(1, 12, negative_slope)
	#self.convs1_siamese[0].weight = self.convs1[0].weight
	
	import numpy as np
	model = UNet3D_MALA().to('cuda:0')
	x = torch.tensor(np.random.random((1, 1, 84, 268, 268)).astype(np.float32)).to('cuda:0')
	out = model(x)
	print(out.shape) # (1, 3, 56, 56, 56)
