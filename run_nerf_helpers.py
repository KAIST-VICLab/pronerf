import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, output_ch-1)
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
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears + 1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear + 1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears + 1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear + 1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear + 1]))


class NeRF_epi(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_epi=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """
        """
        super(NeRF_epi, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_epi = input_ch_epi
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])
        self.epi_linears = nn.ModuleList([nn.Linear(input_ch_epi + W, W // 2), nn.Linear(W // 2, W)])

        # if use_viewdirs:
        #     self.feature_linear = nn.Linear(W, W)
        #     self.alpha_linear = nn.Linear(W, 1)
        #     self.rgb_linear = nn.Linear(W // 2, 3)
        # else:
        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W // 2, output_ch-1)

    def forward(self, x):
        input_pts, input_epi, input_views = \
            torch.split(x, [self.input_ch, self.input_ch_epi, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        h = torch.cat([h, input_epi], -1)
        for i, l in enumerate(self.epi_linears):
            h = self.epi_linears[i](h)
            h = F.relu(h)

        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views], -1)

        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)
        outputs = torch.cat([rgb, alpha], -1)

        return outputs


class NeRF_epi2(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_epi=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """
        """
        super(NeRF_epi2, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_epi = input_ch_epi
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        # Epipolar encoding
        self.epi_linears = nn.ModuleList([nn.Linear(input_ch_epi, W)] +
                                         [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch_epi, W)
                                          for i in range(D//2 - 1)])

        # if use_viewdirs:
        #     self.feature_linear = nn.Linear(W, W)
        #     self.alpha_linear = nn.Linear(W, 1)
        #     self.rgb_linear = nn.Linear(W // 2, 3)
        # else:
        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W // 2, output_ch-1)

    def forward(self, x):
        input_pts, input_epi, input_views = \
            torch.split(x, [self.input_ch, self.input_ch_epi, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        h_pts = h.clone()

        h_e = input_epi
        for i, l in enumerate(self.epi_linears):
            h_e = self.epi_linears[i](h_e)
            h_e = F.relu(h_e)
            if i in self.skips:
                h_e = torch.cat([input_epi, h_e], -1)

        # Outputs with no epi
        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views], -1)

        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)
        outputsA = torch.cat([rgb, alpha], -1)

        # Outputs with epi
        h = (h_pts + h_e) / 2
        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views], -1)

        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)
        outputsB = torch.cat([rgb, alpha], -1)

        return torch.cat((outputsA, outputsB), -1)


class NeRFs(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """
        """
        super(NeRFs, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [SineLayer(input_ch, W)] + [SineLayer(W, W) if i not in self.skips else SineLayer(W + input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([SineLayer(input_ch_views + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
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


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=1.0):
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
        # if not self.print:
        #     print(self.omega_weight_0.detach())
        #     print(self.phase_weight_0.detach())
        #     self.print = True
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

    def __init__(self, in_features, out_features, groups=1, kernel_size=1, padding=0, bias=True,
                 is_first=False, omega_0=1.0):
        super().__init__()
        self.omega_0 = omega_0
        self.omega_weight_0 = torch.nn.Parameter(torch.randn(1, out_features, 1, 1) * 0 + omega_0)
        self.omega_weight_0.requires_grad = True

        self.phase_weight_0 = torch.nn.Parameter(torch.randn(1, out_features, 1, 1) * 0)
        self.phase_weight_0.requires_grad = True

        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Conv2d(in_features, out_features, groups=groups,
                                kernel_size=kernel_size, padding=padding, stride=1, bias=bias)

        self.init_weights()
        self.print = False

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            elif self.omega_0 < 1:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features),
                                            np.sqrt(6 / self.in_features))
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


class MinMaxRay_NetEpi(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_epi=3,output_ch=3, skips=[4]):
        """
        """
        super(MinMaxRay_NetEpi, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips

        self.fc_backbone = nn.ModuleList([nn.Linear(input_ch, W)] +
                                         [nn.Linear(W, W) if i not in self.skips else
                                          nn.Linear(W + input_ch, W) for i in range(D - 1)])

        self.fc_backbone_epi = nn.ModuleList([nn.Linear(input_epi, W)] +
                                         [nn.Linear(W, W) if i not in self.skips else
                                          nn.Linear(W + input_epi, W) for i in range(D - 1)])

        self.fc_output = nn.Linear(W, output_ch)

    def forward(self, x, epi):
        h = x
        for i, l in enumerate(self.fc_backbone):
            h = self.fc_backbone[i](h)
            h = F.elu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)

        h_e = epi
        for i, l in enumerate(self.fc_backbone_epi):
            h_e = self.fc_backbone_epi[i](h_e)
            h_e = F.elu(h_e)
            if i in self.skips:
                h_e = torch.cat([epi, h_e], -1)

        outputs0 = self.fc_output(h)
        outputs1 = self.fc_output((h + h_e) / 2)

        return outputs0, outputs1


class MinMaxRay_NetEpiNPE(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_epi=3,output_ch=3, skips=[4], npe_ch=16):
        """
        """
        super(MinMaxRay_NetEpiNPE, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips

        self.fc_backbone = nn.ModuleList([nn.Linear(input_ch * npe_ch // 3, W)] +
                                         [nn.Linear(W, W) if i not in self.skips else
                                          nn.Linear(W + input_ch * npe_ch // 3, W) for i in range(D - 1)])

        self.fc_backbone_epi = nn.ModuleList([nn.Linear(input_epi, W)] +
                                         [nn.Linear(W, W) if i not in self.skips else
                                          nn.Linear(W + input_epi, W) for i in range(D - 1)])

        self.npe = nn.Sequential(
            nn.Linear(3, npe_ch * 4), nn.ELU(True), nn.Linear(npe_ch * 4, npe_ch), nn.ELU(True)
        )

        self.fc_output = nn.Linear(W, output_ch)

    def forward(self, x, epi):
        n_rays = x.shape[0]
        intx = x.view(-1, 3)
        if n_rays > 1024 * 4:
            x_npe = torch.cat((self.npe(intx[0:intx.shape[0]//2]), self.npe(intx[intx.shape[0]//2::])), 0)
        else:
            x_npe = self.npe(intx)
        x = x_npe.view(n_rays, -1)

        h = x
        for i, l in enumerate(self.fc_backbone):
            h = self.fc_backbone[i](h)
            h = F.elu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)

        h_e = epi
        for i, l in enumerate(self.fc_backbone_epi):
            h_e = self.fc_backbone_epi[i](h_e)
            h_e = F.elu(h_e)
            if i in self.skips:
                h_e = torch.cat([epi, h_e], -1)

        outputs0 = self.fc_output(h)
        outputs1 = self.fc_output((h + h_e) / 2)

        return outputs0, outputs1


class MinMaxRay_NetEpiNPE0(nn.Module):
    def __init__(self, D=8, W=256, input_points=4, input_ch=3, input_epi=3, output_ch=3, skips=[4], npe_ch=16):
        """
        """
        super(MinMaxRay_NetEpiNPE0, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_epi = input_epi
        self.skips = skips
        self.input_points = input_points

        print(f'Input ch: {input_ch}, Input Epi: {input_epi}')

        self.fc_backbone = nn.ModuleList([nn.Linear(input_points * npe_ch, W)] +
                                         [nn.Linear(W, W) if i not in self.skips else
                                          nn.Linear(W + input_points * npe_ch, W) for i in range(D - 1)])

        self.npe = nn.Sequential(
            nn.Linear(input_ch + input_epi, npe_ch * 4), nn.ELU(True), nn.Linear(npe_ch * 4, npe_ch), nn.ELU(True)
        )

        self.fc_output = nn.Linear(W, output_ch)

    def forward(self, x, epi=None, chunk=1024*64):
        n_rays = x.shape[0]
        if epi is not None:
            x = torch.cat((x.view(-1, self.input_ch),  epi.view(-1, self.input_epi)), 1)
        else:
            x = x.view(-1, self.input_ch)

        if x.shape[0] > chunk:
            x_npe = torch.cat([self.npe(x[i:i + chunk]) for i in range(0, x.shape[0], chunk)], 0)
        else:
            x_npe = self.npe(x)
        x = x_npe.view(n_rays, -1)

        h = x
        for i, l in enumerate(self.fc_backbone):
            h = self.fc_backbone[i](h)
            h = F.elu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)

        outputs1 = self.fc_output(h)

        return outputs1


class MinMaxRayS_Net(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=3, skips=[4]):
        """
        """
        super(MinMaxRayS_Net, self).__init__()
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
    def __init__(self, D=8, W=256, in_groups=1, input_ch=3, output_ch=3, skips=[4]):
        """
        """
        super(MinMaxRayS1Conv_Net, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips

        self.fc_backbone = nn.ModuleList([SineLayerC(input_ch, in_groups*W, groups=in_groups, kernel_size=1, padding=0)] +
                                         [SineLayerC(in_groups*W, W, kernel_size=1, padding=0)] +
                                         [SineLayerC(W, W, kernel_size=1) if i not in self.skips else
                                          SineLayerC(W + input_ch, W, kernel_size=1) for i in range(1, D - 1)])

        self.fc_output = nn.Conv2d(W, output_ch, kernel_size=1)

    def forward(self, x):
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


class MinMaxRay_NetCEpi(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=3, input_ch_epi=3, skips=[4]):
        """
        """
        super(MinMaxRay_NetCEpi, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips

        self.fc_backbone = nn.ModuleList([nn.Conv2d(input_ch, W, kernel_size=1)] +
                                         [nn.Conv2d(W, W, kernel_size=1) if i not in self.skips else
                                          nn.Conv2d(W + input_ch, W, kernel_size=1) for i in range(D - 1)])

        self.fc_epi = nn.ModuleList([nn.Conv2d(input_ch_epi + W, W // 2, kernel_size=1),
                                     nn.Conv2d(W // 2, W, kernel_size=1)])

        self.fc_output = nn.Conv2d(W, output_ch,kernel_size=1)

    def forward(self, x, epi):
        h = x
        for i, l in enumerate(self.fc_backbone):
            h = self.fc_backbone[i](h)
            h = F.elu(h)
            if i in self.skips:
                h = torch.cat([x, h], 1)

        h = torch.cat((h, epi), 1)
        for i, l in enumerate(self.fc_epi):
            h = self.fc_epi[i](h)
            h = F.elu(h)

        outputs = self.fc_output(h)

        return outputs


class MinMaxRay_NetCEE(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=3, input_ch_epi=3, skips=[4]):
        """
        """
        super(MinMaxRay_NetCEE, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips

        self.fc_backbone = nn.ModuleList([nn.Conv2d(input_ch, W, kernel_size=1)] +
                                         [nn.Conv2d(W, W, kernel_size=1) if i not in self.skips else
                                          nn.Conv2d(W + input_ch, W, kernel_size=1) for i in range(D - 1)])

        self.fc_epi = nn.ModuleList([nn.Conv2d(input_ch_epi, W, kernel_size=1)] +
                                         [nn.Conv2d(W, W, kernel_size=1) if i not in self.skips else
                                          nn.Conv2d(W + input_ch_epi, W, kernel_size=1) for i in range(D - 1)])

        self.fc_output = nn.Conv2d(W, output_ch,kernel_size=1)

    def forward(self, x, epi):
        h = x
        for i, l in enumerate(self.fc_backbone):
            h = self.fc_backbone[i](h)
            h = F.elu(h)
            if i in self.skips:
                h = torch.cat([x, h], 1)

        h_e = epi
        for i, l in enumerate(self.fc_epi):
            h_e = self.fc_epi[i](h_e)
            h_e = F.elu(h_e)
            if i in self.skips:
                h_e = torch.cat([epi, h_e], 1)

        # outputs = self.fc_output((h + h_e) / 2)
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


def ndc_dist(H, W, focal, near, z_vals):
    # Shift ray origins to near plane
    t = -(near + z_vals)
    z_vals = z_vals + t[..., None]

    # Projection
    z_vals = 1. + 2. * near / z_vals[..., 2]

    return z_vals


def world_2_ndc(H, W, focal, near, world_point):
    # Projection
    o0 = -1. / (W / (2. * focal)) * world_point[..., 0] / world_point[..., 2]
    o1 = -1. / (H / (2. * focal)) * world_point[..., 1] / world_point[..., 2]
    o2 = 1. + 2. * near / world_point[..., 2]

    world_point = torch.stack([o0, o1, o2], -1)

    return world_point


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
