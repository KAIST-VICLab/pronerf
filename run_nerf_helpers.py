import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.model_zoo as model_zoo
import torchvision.models as models


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))

class NeRFTRT(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRFTRT, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D - 1)])

        # Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList(
            [nn.Linear(input_ch_views + W, W // 2)])

        # Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears + 1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear + 1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears + 1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_rbg_linear + 1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_alpha_linear + 1]))

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.omega_weight_0 = torch.nn.Parameter(torch.randn(1, out_features) * 0 + omega_0)
        self.omega_weight_0.requires_grad = True

        self.phase_weight_0 = torch.nn.Parameter(torch.randn(1, out_features) * 0)
        self.phase_weight_0.requires_grad = True

        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()
        self.print = False

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / int(self.omega_0),
                                            np.sqrt(6 / self.in_features) / int(self.omega_0))

    def forward(self, input):
        if not self.print:
            print(self.omega_weight_0.detach())
            print(self.phase_weight_0.detach())
            self.print = True
        return torch.sin(self.omega_weight_0 * self.linear(input) + self.phase_weight_0)

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_weight_0 * self.linear(input) + self.phase_weight_0
        return torch.sin(intermediate), intermediate


class SineLayerC(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, kernel_size=1, padding=0, bias=True,
                 is_first=False, omega_0=1.0):
        super().__init__()
        self.omega_0 = omega_0
        self.omega_weight_0 = torch.nn.Parameter(torch.randn(1, out_features, 1, 1) * 0 + omega_0)
        self.omega_weight_0.requires_grad = True

        self.phase_weight_0 = torch.nn.Parameter(torch.randn(1, out_features, 1, 1) * 0)
        self.phase_weight_0.requires_grad = True

        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Conv2d(in_features, out_features,
                                kernel_size=kernel_size, padding=padding, stride=1, bias=bias)

        self.init_weights()
        self.print = False

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / int(self.omega_0),
                                            np.sqrt(6 / self.in_features) / int(self.omega_0))

    def forward(self, input):
        return torch.sin(self.omega_weight_0 * self.linear(input) + self.phase_weight_0)

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_weight_0 * self.linear(input) + self.phase_weight_0
        return torch.sin(intermediate), intermediate


class MinMaxRay_Net(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=3, skips=[4]):
        """
        """
        super(MinMaxRay_Net, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips

        self.fc_backbone = nn.ModuleList([nn.Linear(input_ch, W)] +
                                         [nn.Linear(W, W) if i not in self.skips else
                                          nn.Linear(W + input_ch, W) for i in range(D - 1)])

        self.fc_output = nn.Linear(W, output_ch)

    def forward(self, x):
        h = x
        for i, l in enumerate(self.fc_backbone):
            h = self.fc_backbone[i](h)
            h = F.elu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)

        outputs = self.fc_output(h)

        return outputs


class MinMaxRayS_Net(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=3, skips=[4]):
        """
        """
        super(MinMaxRayS_Net, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips

        self.fc_backbone = nn.ModuleList([SineLayer(input_ch, W, omega_0=2.0, is_first=True)] +
                                         [SineLayer(W, W, omega_0=1.0) if i not in self.skips else
                                          SineLayer(W + input_ch, W, omega_0=1.0) for i in range(D - 1)])

        self.fc_output = nn.Linear(W, output_ch, bias=False)

    def forward(self, x):
        h = x
        for i, l in enumerate(self.fc_backbone):
            h = self.fc_backbone[i](h)
            if i in self.skips:
                h = torch.cat([x, h], -1)

        outputs = self.fc_output(h)

        return outputs


class MinMaxRayS2_Net(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=3, skips=[4]):
        """
        """
        super(MinMaxRayS2_Net, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips

        self.fc_backbone = nn.ModuleList([SineLayer(input_ch, W, omega_0=1.0, is_first=True)] +
                                         [SineLayer(W, W, omega_0=1.0) if i not in self.skips else
                                          SineLayer(W + input_ch, W, omega_0=1.0) for i in range(D - 1)])

        self.fc_output = nn.Linear(W, output_ch, bias=False)

        self.fc_backbone2 = nn.ModuleList([SineLayer(input_ch + output_ch, W, omega_0=1.0, is_first=True)] +
                                         [SineLayer(W, W, omega_0=1.0) if i not in self.skips else
                                          SineLayer(W + input_ch, W, omega_0=1.0) for i in range(D - 1)])

        self.fc_output2 = nn.Linear(W, output_ch, bias=False)

    def forward(self, x):
        h = x
        for i, l in enumerate(self.fc_backbone):
            h = self.fc_backbone[i](h)
            if i in self.skips:
                h = torch.cat([x, h], -1)
        outputs = self.fc_output(h)

        h = torch.cat([x, outputs], -1)
        for i, l in enumerate(self.fc_backbone):
            h = self.fc_backbone2[i](h)
            if i in self.skips:
                h = torch.cat([x, h], -1)
        outputs2 = self.fc_output2(h)

        return outputs, outputs2


class MinMaxRayS1_Net(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=3, skips=[4]):
        """
        """
        super(MinMaxRayS1_Net, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips

        self.fc_backbone = nn.ModuleList([SineLayer(input_ch, W, omega_0=1.0, is_first=True)] +
                                         [SineLayer(W, W, omega_0=1.0) if i not in self.skips else
                                          SineLayer(W + input_ch, W, omega_0=1.0) for i in range(D - 1)])

        self.fc_output = nn.Linear(W, output_ch, bias=False)

    def forward(self, x):
        h = x
        for i, l in enumerate(self.fc_backbone):
            h = self.fc_backbone[i](h)
            if i in self.skips:
                h = torch.cat([x, h], -1)
        outputs = self.fc_output(h)

        return outputs


class MinMaxRayS1Conv_Net(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=3, skips=[4]):
        """
        """
        super(MinMaxRayS1Conv_Net, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips

        self.fc_backbone = nn.ModuleList([SineLayerC(input_ch, W, kernel_size=1, padding=0)] +
                                         [SineLayerC(W, W, kernel_size=1) if i not in self.skips else
                                          SineLayerC(W + input_ch, W, kernel_size=1) for i in range(D - 1)])

        self.fc_output = nn.Conv2d(W, output_ch,kernel_size=1)

    def forward(self, x):
        h = x
        for i, l in enumerate(self.fc_backbone):
            h = self.fc_backbone[i](h)
            h = F.elu(h)
            if i in self.skips:
                h = torch.cat([x, h], 1)

        outputs = self.fc_output(h)

        return outputs

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x

class MinMaxRayS1ConvRes_Net(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, res_ch=3, output_ch=3, skips=[4]):
        """
        """
        super(MinMaxRayS1ConvRes_Net, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.unet_out = 64

        self.fc_backbone = nn.ModuleList([SineLayerC(input_ch + self.unet_out, W, kernel_size=1, padding=0)] +
                                         [SineLayerC(W, W, kernel_size=1) if i not in self.skips else
                                          SineLayerC(W + input_ch + self.unet_out, W, kernel_size=1) for i in range(D - 1)])

        self.fc_output = nn.Conv2d(W, output_ch,kernel_size=1)

        self.encoder = UNet(res_ch)

    def forward(self, x, warp_x):
        warp_feat = self.encoder(warp_x)
        x = torch.cat([x,warp_feat], dim=1)
        h = x
        for i, l in enumerate(self.fc_backbone):
            h = self.fc_backbone[i](h)
            h = F.elu(h)
            if i in self.skips:
                h = torch.cat([x, h], 1)

        outputs = self.fc_output(h)

        return outputs


class MinMaxRayS15_Net(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=3, skips=[4]):
        """
        """
        super(MinMaxRayS15_Net, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips

        self.fc_backbone = nn.ModuleList([nn.Linear(input_ch, W)] +
                                         [nn.Linear(W, W) if i not in self.skips else
                                          nn.Linear(W + input_ch, W) for i in range(D - 1)])
        self.fc_output = nn.Linear(W, output_ch, bias=False)

        self.fc_backbone2 = nn.ModuleList([SineLayer(input_ch + output_ch, W, omega_0=3.0, is_first=True)] +
                                         [SineLayer(W, W, omega_0=1.0) if i not in self.skips else
                                          SineLayer(W + input_ch, W, omega_0=1.0) for i in range(D - 1)])
        self.fc_output2 = nn.Linear(W, output_ch, bias=False)

    def forward(self, x):
        h = x
        for i, l in enumerate(self.fc_backbone):
            h = self.fc_backbone[i](h)
            h = F.elu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)
        outputs = self.fc_output(h)

        h = torch.cat([x, outputs], -1)
        for i, l in enumerate(self.fc_backbone):
            h = self.fc_backbone2[i](h)
            if i in self.skips:
                h = torch.cat([x, h], -1)
        outputs2 = self.fc_output2(h)

        return outputs, outputs2


class MinMaxRay2_Net(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=3, skips=[4]):
        """
        """
        super(MinMaxRay2_Net, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips

        self.fc_backbone = nn.ModuleList([nn.Linear(input_ch, W)] +
                                         [nn.Linear(W, W) if i not in self.skips else
                                          nn.Linear(W + input_ch, W) for i in range(D - 1)])
        self.fc_output = nn.Linear(W, output_ch, bias=False)

        self.fc_backbone2 = nn.ModuleList([nn.Linear(input_ch + output_ch, W)] +
                                         [nn.Linear(W, W) if i not in self.skips else
                                          nn.Linear(W + input_ch, W) for i in range(D - 1)])
        self.fc_output2 = nn.Linear(W, output_ch, bias=False)

    def forward(self, x):
        h = x
        for i, l in enumerate(self.fc_backbone):
            h = self.fc_backbone[i](h)
            h = F.elu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)
        outputs = self.fc_output(h)

        h = torch.cat([x, outputs], -1)
        for i, l in enumerate(self.fc_backbone):
            h = self.fc_backbone2[i](h)
            h = F.elu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)
        outputs2 = self.fc_output2(h)

        return outputs, outputs2


class MinMaxRay_NetConv(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=3, skips=[4]):
        """
        """
        super(MinMaxRay_NetConv, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips

        self.fc_backbone = nn.ModuleList([nn.Conv2d(input_ch, W, kernel_size=1)] +
                                         [nn.Conv2d(W, W, kernel_size=1) if i not in self.skips else
                                          nn.Conv2d(W + input_ch, W, kernel_size=1) for i in range(D - 1)])

        self.fc_output = nn.Conv2d(W, output_ch,kernel_size=1)

    def forward(self, x):
        h = x
        for i, l in enumerate(self.fc_backbone):
            h = self.fc_backbone[i](h)
            h = F.elu(h)
            if i in self.skips:
                h = torch.cat([x, h], 1)

        outputs = self.fc_output(h)

        return outputs

# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def get_centered_rays(H, W, K, c2w):
    dirs = torch.Tensor([0, 0, -1], device = c2w.device)[None]
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:,:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:,:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d

def get_world_from_ndc(H, W, focal, points, near=1.0):
    # ndc_z = 1+2n/z => z =...
    oz = (2 * near) / (points[:, -1] - 1)
    # ndc x = (-2f/W)*(x/z) ==>
    ox = (points[:, 0] * oz) / (-(2 * focal) / W)
    oy = (points[:, 1] * oz) / (-(2 * focal) / H)
    return torch.stack([ox, oy, oz], dim=-1)

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
