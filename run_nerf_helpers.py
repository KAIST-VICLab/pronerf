import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import torch.nn.init as init
from kornia.losses import ssim_loss as dssim


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log10(x)
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def img2ssim(image_pred, image_gt, reduction="mean"):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    dssim_ = dssim(image_pred, image_gt, 3, reduction=reduction)  # dissimilarity in [0, 1]
    return 1 - 2 * dssim_  # in [-1, 1]

def compute_angle(xyz, query_camera, train_cameras):
    """
    :param xyz: [nrays,nsamples, 3]
    :param query_camera: [nrays,3,4]
    :param train_cameras: [nrays, nviews,3,4]
    :return: [n_views, ..., 4]; The first 3 channels are unit-length vector of the difference between
    query and target ray directions, the last channel is the inner product of the two directions.
    """
    n_rays, n_views,_,_ = train_cameras.shape
    _, n_samples,_ = xyz.shape
    ray2tar_pose = query_camera[:, :3, 3].unsqueeze(1) - xyz # nrays, nsamples, 3
    ray2tar_pose = ray2tar_pose/torch.norm(ray2tar_pose, dim=-1, keepdim=True) + 1e-6

    ray2train_pose = train_cameras[:, :, :3, 3].unsqueeze(1) - xyz.unsqueeze(2)
    ray2train_pose = ray2train_pose/torch.norm(ray2train_pose, dim=-1, keepdim=True) + 1e-6

    ray_diff = ray2tar_pose.unsqueeze(2) - ray2train_pose
    ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)
    ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-6)

    ray_diff_dot = torch.sum(ray2tar_pose.unsqueeze(2) * ray2train_pose, dim=-1, keepdim=True)
    ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1) # nrays, nsamples, nviews, 4
    return ray_diff


class Pluecker(nn.Module):
    def __init__(
        self,
    ):

        super().__init__()

        self.in_channels = 6
        self.out_channels = 6

        self.direction_multiplier = 1.0
        self.moment_multiplier =  1.0

        self.origin = torch.tensor([0.0, 0.0, 0.0], device='cuda')

    def forward(self, rays_o, rays_d):
        rays_d = torch.nn.functional.normalize(rays_d, p=2.0, dim=-1)

        m = torch.cross(rays_o, rays_d, dim=-1)
        return torch.cat([rays_d * self.direction_multiplier, m * self.moment_multiplier], dim=-1)

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
            x = torch.cat((x.view(-1, self.input_ch),  epi.reshape(-1, self.input_epi)), 1)
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

class MinMaxRayAttn_Net(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, pos_enc=None,view_enc=None,posenc_dim=3, viewenc_dim=3, output_ch=3):
        """
        """
        super(MinMaxRayAttn_Net, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.posenc_dim = posenc_dim
        self.viewenc_dim = viewenc_dim

        # input project
        self.rgbfeat_fc = nn.Sequential(
            nn.Linear(input_ch, W),
            nn.ReLU(),
            nn.Linear(W, W),
        )

        self.view_selftrans = nn.ModuleList([])
        self.view_crosstrans = nn.ModuleList([])
        self.q_fcs = nn.ModuleList([])

        for i in range(D):
            # view transformer
            view_trans = Transformer2D(
                dim=W,
                ff_hid_dim=int(W * 4),
                ff_dp_rate=0.1,
                attn_dp_rate=0.1,
            )
            self.view_crosstrans.append(view_trans)

            # ray transformer
            ray_trans = Transformer(
                dim=W,
                ff_hid_dim=int(W * 4),
                n_heads=4,
                ff_dp_rate=0.1,
                attn_dp_rate=0.1,
            )
            self.view_selftrans.append(ray_trans)

            # mlp
            if i % 2 == 0:
                q_fc = nn.Sequential(
                    nn.Linear(W + posenc_dim + viewenc_dim, W),
                    nn.ReLU(),
                    nn.Linear(W, W),
                )
            else:
                q_fc = nn.Identity()
            self.q_fcs.append(q_fc)

        self.norm = nn.LayerNorm(W)
        self.outout_fc = nn.Linear(W, output_ch)
        self.pos_enc = pos_enc
        self.view_enc = view_enc

    def forward(self, rgb_feat, ray_diff, mask, pts, ray_d):
        viewdirs = ray_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
        viewdirs = self.view_enc(viewdirs)

        pts_ = torch.reshape(pts, [-1, pts.shape[-1]]).float()
        pts_ = self.pos_enc(pts_)
        pts_ = torch.reshape(pts_, list(pts.shape[:-1]) + [pts_.shape[-1]])
        viewdirs_ = viewdirs[:, None].repeat(1, pts_.shape[1], 1)
        embed = torch.cat([pts_, viewdirs_], dim=-1)
        input_pts, input_views = torch.split(embed, [self.posenc_dim, self.viewenc_dim], dim=-1)
        
        rgb_feat = self.rgbfeat_fc(rgb_feat)
        feat_query = rgb_feat.max(dim=2)[0]
        for i, (crosstrans, q_fc, selftrans) in enumerate(
            zip(self.view_crosstrans, self.q_fcs, self.view_selftrans)
        ):
            feat_query = crosstrans(feat_query, rgb_feat, ray_diff, mask)
            if i % 2 == 0:
                feat_query = torch.cat((feat_query, input_pts, input_views), dim=-1)
                feat_query = q_fc(feat_query)

            # ray transformer
            feat_query = selftrans(feat_query, ret_attn=False)

        # normalize & rgb
        h = self.norm(feat_query)
        outputs = self.outout_fc(h.mean(dim=1))
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

class MinMaxRaySOrder_Net(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, N_samples=3, skips=[4]):
        """
        """
        super(MinMaxRaySOrder_Net, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.near = 0.
        self.far = 1.
        self.N_samples = N_samples

        self.fc_backbone = nn.ModuleList([SineLayer(input_ch, W, omega_0=2.0, is_first=True)] +
                                         [SineLayer(W, W, omega_0=1.0) if i not in self.skips else
                                          SineLayer(W + input_ch, W, omega_0=1.0) for i in range(D - 1)])

        self.fc_output = nn.Linear(W, N_samples*2 + 3, bias=False)
        self.pos_output = nn.Linear(W, N_samples, bias=False)

    # def forward(self, x):
    #     h = x
    #     for i, l in enumerate(self.fc_backbone):
    #         h = self.fc_backbone[i](h)
    #         if i in self.skips:
    #             h = torch.cat([x, h], -1)

    #     dense_outputs = self.fc_output(h)
    #     pos_outputs = self.pos_output(h)
    #     pos_lists = []
    #     for i in range(self.N_samples):
    #         if i == 0:
    #             pos_lists.append((self.far - self.near) * torch.sigmoid(pos_outputs[:, i,None]) + self.near)
    #         else:
    #             pos_lists.append(torch.sigmoid(pos_outputs[:, i,None]) * (self.far - pos_lists[i-1]) +  pos_lists[i-1])
    #     pos_lists = torch.cat(pos_lists, dim =1)
    #     outputs = torch.cat([pos_lists, dense_outputs], dim =1)
    #     return outputs

    def forward(self, x):
        h = x
        for i, l in enumerate(self.fc_backbone):
            h = self.fc_backbone[i](h)
            if i in self.skips:
                h = torch.cat([x, h], -1)

        dense_outputs = self.fc_output(h)
        pos_outputs = self.pos_output(h)
        pos_lists = []
        for i in range(self.N_samples):
            if i == 0:
                pos_lists.append((self.far - self.near) * (1-torch.sigmoid(pos_outputs[:, i,None])) + self.near)
            else:
                pos_lists.append((1-torch.sigmoid(pos_outputs[:, i,None])) * (self.far - pos_lists[i-1]) +  pos_lists[i-1])
        pos_lists = torch.cat(pos_lists, dim =1)
        outputs = torch.cat([pos_lists, dense_outputs], dim =1)
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib


def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    return getattr(m, class_name)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
        padding_mode="reflect",
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False, padding_mode="reflect"
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, track_running_stats=False, affine=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width, track_running_stats=False, affine=True)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width, track_running_stats=False, affine=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            num_in_layers,
            num_out_layers,
            kernel_size=kernel_size,
            stride=stride,
            padding=(self.kernel_size - 1) // 2,
            padding_mode="reflect",
        )
        self.bn = nn.InstanceNorm2d(num_out_layers, track_running_stats=False, affine=True)

    def forward(self, x):
        return F.elu(self.bn(self.conv(x)), inplace=True)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(
            x, scale_factor=self.scale, align_corners=True, mode="bilinear"
        )
        return self.conv(x)


class ResUNet(nn.Module):
    def __init__(
        self,
        encoder="resnet34",
        coarse_out_ch=32,
        fine_out_ch=32,
        norm_layer=None,
        single_net=True,
    ):

        super(ResUNet, self).__init__()
        assert encoder in [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
        ], "Incorrect encoder type"
        if encoder in ["resnet18", "resnet34"]:
            filters = [64, 128, 256, 512]
        else:
            filters = [256, 512, 1024, 2048]
        self.single_net = single_net
        if self.single_net:
            self.coarse_out_ch = coarse_out_ch
            self.fine_out_ch = coarse_out_ch
            out_ch = coarse_out_ch
        else:
            self.coarse_out_ch = coarse_out_ch
            self.fine_out_ch = fine_out_ch
            out_ch = coarse_out_ch + fine_out_ch

        # original
        layers = [3, 4, 6, 3]
        if norm_layer is None:
            # norm_layer = nn.BatchNorm2d
            norm_layer = nn.InstanceNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1
        block = BasicBlock
        replace_stride_with_dilation = [False, False, False]
        self.inplanes = 64
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(
            3,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            padding_mode="reflect",
        )
        self.bn1 = norm_layer(self.inplanes, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )

        # decoder
        self.upconv3 = upconv(filters[2], 128, 3, 2)
        self.iconv3 = conv(filters[1] + 128, 128, 3, 1)
        self.upconv2 = upconv(128, 64, 3, 2)
        self.iconv2 = conv(filters[0] + 64, out_ch, 3, 1)

        # fine-level conv
        self.out_conv = nn.Conv2d(out_ch, out_ch, 1, 1)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, track_running_stats=False, affine=True),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def skipconnect(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = self.upconv3(x3)
        x = self.skipconnect(x2, x)
        x = self.iconv3(x)

        x = self.upconv2(x)
        x = self.skipconnect(x1, x)
        x = self.iconv2(x)

        x_out = self.out_conv(x)
        if self.single_net:
            x_coarse = x_out
            x_fine = x_out
        else:
            x_coarse = x_out[:, : self.coarse_out_ch, :]
            x_fine = x_out[:, -self.fine_out_ch :, :]
        return x_coarse, x_fine

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
    
class FeedForward(nn.Module):
    def __init__(self, dim, hid_dim, dp_rate):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.dp(self.activ(self.fc1(x)))
        x = self.dp(self.fc2(x))
        return x


# Subtraction-based efficient attention
class Attention2D(nn.Module):
    def __init__(self, dim, dp_rate):
        super(Attention2D, self).__init__()
        self.q_fc = nn.Linear(dim, dim, bias=False)
        self.k_fc = nn.Linear(dim, dim, bias=False)
        self.v_fc = nn.Linear(dim, dim, bias=False)
        self.pos_fc = nn.Sequential(
            nn.Linear(4, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        self.attn_fc = nn.Sequential(
            nn.Linear(dim, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        self.out_fc = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp_rate)

    def forward(self, q, k, pos, mask=None):
        q = self.q_fc(q)
        k = self.k_fc(k)
        v = self.v_fc(k)

        pos = self.pos_fc(pos)
        attn = k - q[:, :, None, :] + pos
        attn = self.attn_fc(attn)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn, dim=-2)
        attn = self.dp(attn)

        x = ((v + pos) * attn).sum(dim=2)
        x = self.dp(self.out_fc(x))
        return x


# View Transformer
class Transformer2D(nn.Module):
    def __init__(self, dim, ff_hid_dim, ff_dp_rate, attn_dp_rate):
        super(Transformer2D, self).__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6)

        self.ff = FeedForward(dim, ff_hid_dim, ff_dp_rate)
        self.attn = Attention2D(dim, attn_dp_rate)

    def forward(self, q, k, pos, mask=None):
        residue = q
        x = self.attn_norm(q)
        x = self.attn(x, k, pos, mask)
        x = x + residue

        residue = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = x + residue

        return x


# attention module for self attention.
# contains several adaptations to incorportate positional information (NOT IN PAPER)
#   - qk (default) -> only (q.k) attention.
#   - pos -> replace (q.k) attention with position attention.
#   - gate -> weighted addition of  (q.k) attention and position attention.
class Attention(nn.Module):
    def __init__(self, dim, n_heads, dp_rate, attn_mode="qk", pos_dim=None):
        super(Attention, self).__init__()
        if attn_mode in ["qk", "gate"]:
            self.q_fc = nn.Linear(dim, dim, bias=False)
            self.k_fc = nn.Linear(dim, dim, bias=False)
        if attn_mode in ["pos", "gate"]:
            self.pos_fc = nn.Sequential(
                nn.Linear(pos_dim, pos_dim), nn.ReLU(), nn.Linear(pos_dim, dim // 8)
            )
            self.head_fc = nn.Linear(dim // 8, n_heads)
        if attn_mode == "gate":
            self.gate = nn.Parameter(torch.ones(n_heads))
        self.v_fc = nn.Linear(dim, dim, bias=False)
        self.out_fc = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.n_heads = n_heads
        self.attn_mode = attn_mode

    def forward(self, x, pos=None, ret_attn=False):
        if self.attn_mode in ["qk", "gate"]:
            q = self.q_fc(x)
            q = q.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)
            k = self.k_fc(x)
            k = k.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)
        v = self.v_fc(x)
        v = v.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)

        if self.attn_mode in ["qk", "gate"]:
            attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.shape[-1])
            attn = torch.softmax(attn, dim=-1)
        elif self.attn_mode == "pos":
            pos = self.pos_fc(pos)
            attn = self.head_fc(pos[:, :, None, :] - pos[:, None, :, :]).permute(0, 3, 1, 2)
            attn = torch.softmax(attn, dim=-1)
        if self.attn_mode == "gate":
            pos = self.pos_fc(pos)
            pos_attn = self.head_fc(pos[:, :, None, :] - pos[:, None, :, :]).permute(0, 3, 1, 2)
            pos_attn = torch.softmax(pos_attn, dim=-1)
            gate = self.gate.view(1, -1, 1, 1)
            attn = (1.0 - torch.sigmoid(gate)) * attn + torch.sigmoid(gate) * pos_attn
            attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn = self.dp(attn)

        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous()
        out = out.view(x.shape[0], x.shape[1], -1)
        out = self.dp(self.out_fc(out))
        if ret_attn:
            return out, attn
        else:
            return out


# Ray Transformer
class Transformer(nn.Module):
    def __init__(
        self, dim, ff_hid_dim, ff_dp_rate, n_heads, attn_dp_rate, attn_mode="qk", pos_dim=None
    ):
        super(Transformer, self).__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6)

        self.ff = FeedForward(dim, ff_hid_dim, ff_dp_rate)
        self.attn = Attention(dim, n_heads, attn_dp_rate, attn_mode, pos_dim)

    def forward(self, x, pos=None, ret_attn=False):
        residue = x
        x = self.attn_norm(x)
        x = self.attn(x, pos, ret_attn)
        if ret_attn:
            x, attn = x
        x = x + residue

        residue = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = x + residue

        if ret_attn:
            return x, attn.mean(dim=1)[:, 0]
        else:
            return x

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

