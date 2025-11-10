# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import numpy as np

from lpips import LPIPS
from pytorch_msssim import SSIM
from fused_ssim import fused_ssim
import torch.nn.functional as F

metric_networks = {}


def l1_loss(x, y):
    return torch.nn.functional.l1_loss(x, y)

def l2_loss(x, y):
    return torch.nn.functional.mse_loss(x, y)

def huber_loss(x, y, thres=0.01):
    l1 = (x - y).abs().mean(0)
    l2 = (x - y).pow(2).mean(0)
    loss = torch.where(
        l1 < thres,
        l2,
        2 * thres * l1 - thres ** 2)
    return loss.mean()

def cauchy_loss(x, y, reduction='mean'):
    loss_map = torch.log1p(torch.square(x - y))
    if reduction == 'sum':
        return loss_map.sum()
    if reduction == 'mean':
        return loss_map.mean()
    raise NotImplementedError

def psnr_score(x, y):
    return -10 * torch.log10(l2_loss(x, y))

def ssim_score(x, y):
    if 'SSIM' not in metric_networks:
        metric_networks['SSIM'] = SSIM(data_range=1, win_size=11, win_sigma=1.5, channel=3).cuda()
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    if len(y.shape) == 3:
        y = y.unsqueeze(0)
    return metric_networks['SSIM'](x, y)

def ssim_loss(x, y):
    return 1 - ssim_score(x, y)

def fast_ssim_loss(x, y):
    # Note! Only x get gradient in backward.
    is_train = x.requires_grad or y.requires_grad
    return 1 - fused_ssim(x.unsqueeze(0), y.unsqueeze(0), padding="valid", train=is_train)

def lpips_loss(x, y, net='vgg'):
    key = f'LPIPS_{net}'
    if key not in metric_networks:
        metric_networks[key] = LPIPS(net=net, version='0.1').cuda()
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    if len(y.shape) == 3:
        y = y.unsqueeze(0)
    return metric_networks[key](x, y)

def correct_lpips_loss(x, y, net='vgg'):
    key = f'LPIPS_{net}'
    if key not in metric_networks:
        metric_networks[key] = LPIPS(net=net, version='0.1').cuda()
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    if len(y.shape) == 3:
        y = y.unsqueeze(0)
    return metric_networks[key](x*2-1, y*2-1)

def entropy_loss(prob):
    pos_prob = prob.clamp(1e-6, 1-1e-6)
    neg_prob = 1 - pos_prob
    return -(pos_prob * pos_prob.log() + neg_prob * neg_prob.log()).mean()

def prob_concen_loss(prob):
    return (prob.square() * (1 - prob).square()).mean()


def exp_anneal(end_mul, iter_now, iter_from, iter_end):
    if end_mul == 1 or iter_now >= iter_end:
        return 1
    total_len = iter_end - iter_from + 1
    now_len = max(0, iter_now - iter_from + 1)
    now_p = min(1.0, now_len / total_len)
    return end_mul ** now_p


class SparseDepthLoss:
    def __init__(self, iter_end):
        self.iter_end = iter_end

    def is_active(self, iteration):
        return iteration <= self.iter_end

    def __call__(self, cam, render_pkg):
        assert "raw_T" in render_pkg, "Forgot to set `output_depth=True` when calling render?"
        assert "raw_depth" in render_pkg, "Forgot to set `output_depth=True` when calling render?"
        assert hasattr(cam, "sparse_pt") and cam.sparse_pt is not None, "No sparse points depth?"
        depth = render_pkg['raw_depth'][0] / (1 - render_pkg['raw_T']).clamp_min_(1e-4)
        sparse_pt = cam.sparse_pt.cuda()
        sparse_uv, sparse_depth = cam.project(sparse_pt, return_depth=True)
        rend_sparse_depth = torch.nn.functional.grid_sample(
            depth[None],
            sparse_uv[None,None],
            mode='bilinear', align_corners=False).squeeze()
        sparse_depth = sparse_depth.squeeze(1)
        return torch.nn.functional.smooth_l1_loss(rend_sparse_depth, sparse_depth)

'''
class DepthAnythingv2Loss:
    def __init__(self, iter_from, iter_end, end_mult):
        self.iter_from = iter_from
        self.iter_end = iter_end
        self.end_mult = end_mult

    def is_active(self, iteration):
        return iteration >= self.iter_from and iteration <= self.iter_end

    def __call__(self, cam, render_pkg, iteration):
        assert hasattr(cam, "depthanythingv2"), "Estimated depth not loaded"
        assert "raw_T" in render_pkg, "Forgot to set `output_depth=True` when calling render?"
        assert "raw_depth" in render_pkg, "Forgot to set `output_depth=True` when calling render?"

        if not self.is_active(iteration):
            return 0

        invdepth = 1 / render_pkg['raw_depth'].unsqueeze(1).clamp_min(cam.near)
        alpha = (1 - render_pkg['raw_T'][None])
        mono = cam.depthanythingv2.cuda()
        mono = mono[None,None]

        if invdepth.shape[-2:] != mono.shape[-2:]:
            mono = torch.nn.functional.interpolate(
                mono, size=invdepth.shape[-2:], mode='bilinear')

        X, _, Xref, _= invdepth.split(1)
        X = X * alpha
        Y = mono

        with torch.no_grad():
            Ymed = Y.median()
            Ys = (Y - Ymed).abs().mean()
            Xmed = Xref.median()
            Xs = (Xref - Xmed).abs().mean()
            target = (Y - Ymed) * (Xs/Ys) + Xmed

        mask = (target > 0.01) & (alpha > 0.5)
        X = X * mask
        target = target * mask
        loss = l2_loss(X, target)

        ratio = (iteration - self.iter_from) / (self.iter_end - self.iter_from)
        mult = self.end_mult ** ratio
        return mult * loss
'''

class DepthAnythingv2Loss:
    def __init__(self, iter_from, iter_end, end_mult):
        self.iter_from = iter_from
        self.iter_end = iter_end
        self.end_mult = end_mult

    def is_active(self, iteration):
        return iteration >= self.iter_from and iteration <= self.iter_end
    def __call__(self, cam, render_pkg, iteration):
        assert hasattr(cam, "depthanythingv2"), "Estimated depth not loaded"
        assert "raw_T" in render_pkg, "Forgot to set `output_depth=True` when calling render?"
        assert "raw_depth" in render_pkg, "Forgot to set `output_depth=True` when calling render?"

        if not self.is_active(iteration):
            return 0.0

        # --- 1. 데이터 준비 (원본 코드 구조 유지) ---
        invdepth = 1 / render_pkg['raw_depth'].unsqueeze(1).clamp_min(cam.near)
        
        # alpha는 [1, 1, H, W] shape를 가질 것으로 예상 (원본 코드 기준)
        alpha = (1 - render_pkg['raw_T'][None]) 
        
        # mono는 [1, 1, H, W] shape를 가질 것으로 예상
        mono = cam.depthanythingv2.cuda()
        mono = mono[None,None]

        if invdepth.shape[-2:] != mono.shape[-2:]:
            mono = torch.nn.functional.interpolate(
                mono, size=invdepth.shape[-2:], mode='bilinear', align_corners=False)

        # (수정) 원본의 split(1) 로직을 복원합니다.
        # invdepth는 [4, 1, H, W] -> X, Xref 등은 [1, 1, H, W]
        try:
            X, _, Xref, _ = invdepth.split(1)
        except ValueError:
            # 배치 크기가 4가 아닐 경우를 대비 (예: 배치 1)
            X = invdepth
            Xref = invdepth

        # X: 손실 계산에 사용할 렌더링 깊이 [1, 1, H, W]
        # Xref: 정렬(alignment)에 사용할 렌더링 깊이 [1, 1, H, W]
        # Y: Mono 깊이 [1, 1, H, W]
        Y = mono 

        # --- 2. (수정) 요청하신 '선(先) 마스킹, 후(後) 정렬' ---
        with torch.no_grad():
            # (수정) 정렬(alignment)을 위한 마스크를 먼저 정의합니다.
            # 님의 의도대로 '물체가 있는 곳(alpha > 0.5)'만 사용합니다.
            # 또한 정렬에 사용할 Y(mono)와 Xref(rendered) 값 자체도 유효해야 합니다.
            align_mask = (alpha > 0.5) & (Y > 0.01) & (Xref > 0.01)

            # 유효 픽셀이 하나도 없으면 손실 0 반환 (에러 방지)
            if not align_mask.any():
                return 0.0

            # (수정) 마스크된 영역의 값만으로 통계치를 계산합니다.
            Y_masked = Y[align_mask]
            Xref_masked = Xref[align_mask] # 원본과 동일하게 Xref 사용

            Ymed = Y_masked.median()
            Ys = (Y_masked - Ymed).abs().mean()
            Xmed = Xref_masked.median() # Xref의 중간값
            Xs = (Xref_masked - Xmed).abs().mean() # Xref의 스케일

            if Ys < 1e-6: Ys = 1e-6

            # Y(mono)를 Xref(rendered ref)의 스케일과 시프트에 맞게 정렬
            target = (Y - Ymed) * (Xs / Ys) + Xmed

        # --- 3. 손실 계산 (원본 코드 로직 유지) ---
        
        # 원본 코드는 X(첫 번째 이미지)와 alpha를 곱해서 사용했습니다.
        X = X * alpha 
        
        # 최종 손실 계산을 위한 마스크
        loss_mask = (target > 0.01) & (alpha > 0.5)

        if not loss_mask.any():
            return 0.0
            
        X = X * loss_mask
        target = target * loss_mask
        
        # l2_loss가 (A[mask] - B[mask]).pow(2).mean() 형태가 아니라면,
        # loss = l2_loss(X[loss_mask], target[loss_mask]) 로 변경해야 할 수 있습니다.
        # 일단 원본의 마스크 곱셈 방식을 유지합니다.
        loss = l2_loss(X, target)

        # --- 4. 손실 가중치 적용 ---
        ratio = (iteration - self.iter_from) / (self.iter_end - self.iter_from)
        mult = self.end_mult ** ratio
        
        return mult * loss


class Mast3rMetricDepthLoss:
    def __init__(self, iter_from, iter_end, end_mult):
        self.iter_from = iter_from
        self.iter_end = iter_end
        self.end_mult = end_mult

    def is_active(self, iteration):
        return iteration >= self.iter_from and iteration <= self.iter_end

    def __call__(self, cam, render_pkg, iteration):
        assert hasattr(cam, "mast3r_metric_depth"), "Estimated depth not loaded"
        assert "raw_T" in render_pkg, "Forgot to set `output_depth=True` when calling render?"
        assert "raw_depth" in render_pkg, "Forgot to set `output_depth=True` when calling render?"

        if not self.is_active(iteration):
            return 0

        alpha = (1 - render_pkg['raw_T'][None])
        depth = render_pkg['raw_depth'][[0]][None]
        ref = cam.mast3r_metric_depth[None,None].cuda()

        if depth.shape[-2:] != ref.shape[-2:]:
            alpha = torch.nn.functional.interpolate(
                alpha, size=ref.shape[-2:], mode='bilinear', antialias=True)
            depth = torch.nn.functional.interpolate(
                depth, size=ref.shape[-2:], mode='bilinear', antialias=True)
            # ref = torch.nn.functional.interpolate(
            #     ref, size=depth.shape[-2:], mode='bilinear')

        # Compute cauchy loss
        active_idx = torch.where(alpha > 0.5)
        depth = depth / alpha
        loss = cauchy_loss(depth[active_idx], ref[active_idx], reduction='sum')
        loss = loss * (1 / depth.numel())

        ratio = (iteration - self.iter_from) / (self.iter_end - self.iter_from)
        mult = self.end_mult ** ratio
        return mult * loss

'''
class NormalDepthConsistencyLoss:
    def __init__(self, iter_from, iter_end, ks, tol_deg):
        self.iter_from = iter_from
        self.iter_end = iter_end
        self.ks = ks
        self.tol_cos = np.cos(np.deg2rad(tol_deg))

    def is_active(self, iteration):
        return iteration >= self.iter_from and iteration <= self.iter_end

    def __call__(self, cam, render_pkg, iteration):
        assert "raw_T" in render_pkg, "Forgot to set `output_T=True` when calling render?"
        assert "raw_depth" in render_pkg, "Forgot to set `output_depth=True` when calling render?"
        assert "raw_normal" in render_pkg, "Forgot to set `output_normal=True` when calling render?"

        if not self.is_active(iteration):
            return 0

        # Read rendering results
        render_alpha = 1 - render_pkg['raw_T'].detach().squeeze(0)
        render_depth = render_pkg['raw_depth'][0]
        render_med = render_pkg['raw_depth'][2]
        render_normal = render_pkg['raw_normal']

        # Compute depth to normal
        N_mean = cam.depth2normal(render_depth, ks=self.ks, tol_cos=self.tol_cos)

        # Blend with alpha and compute target
        target = render_alpha.square()
        N_mean = N_mean * render_alpha

        # Compute loss
        mask = (N_mean != 0).any(0)
        #loss_map = (target - (render_normal * N_mean).sum(dim=0)) * mask
        diff = torch.square(render_med - render_depth)  # L1
        # 유효 픽셀만 평균 (0으로 나눔 방지)
        num = mask.sum().clamp_min(1)
        loss = (diff * mask.float()).sum() / num
        return loss
'''
class NormalDepthConsistencyLoss:
    def __init__(self, iter_from, iter_end, ks, tol_deg):
        self.iter_from = iter_from
        self.iter_end = iter_end
        self.ks = ks
        self.tol_cos = np.cos(np.deg2rad(tol_deg))

    def is_active(self, iteration):
        return iteration >= self.iter_from and iteration <= self.iter_end

    def __call__(self, cam, render_pkg, iteration):
        assert "raw_T" in render_pkg, "Forgot to set `output_T=True` when calling render?"
        assert "raw_depth" in render_pkg, "Forgot to set `output_depth=True` when calling render?"
        assert "raw_normal" in render_pkg, "Forgot to set `output_normal=True` when calling render?"

        if not self.is_active(iteration):
            return 0

        # Read rendering results
        render_alpha = 1 - render_pkg['raw_T'].detach().squeeze(0)
        render_depth = render_pkg['raw_depth'][0]
        render_normal = render_pkg['raw_normal']

        # Compute depth to normal
        N_mean = -cam.depth2normal(render_depth, ks=self.ks, tol_cos=self.tol_cos)

        # Blend with alpha and compute target
        target = render_alpha.square()
        N_mean = N_mean * render_alpha

        # Compute loss
        mask = (N_mean != 0).any(0)
        loss_map = (target - (render_normal * N_mean).sum(dim=0)) * mask
        loss = loss_map.mean()
        return loss

class NormalMedianConsistencyLoss:
    def __init__(self, iter_from, iter_end):
        self.iter_from = iter_from
        self.iter_end = iter_end

    def is_active(self, iteration):
        return iteration >= self.iter_from and iteration <= self.iter_end

    def __call__(self, cam, render_pkg, iteration):
        assert "raw_depth" in render_pkg, "Forgot to set `output_depth=True` when calling render?"
        assert "raw_normal" in render_pkg, "Forgot to set `output_normal=True` when calling render?"

        if not self.is_active(iteration):
            return 0

        # TODO: median depth is not differentiable
        render_median = render_pkg['raw_depth'][2]
        render_normal = render_pkg['raw_normal']

        # Compute depth to normal
        N_med = -cam.depth2normal(render_median, ks=3)

        # Compute loss
        mask = (N_med != 0).any(0)
        loss_map = (1 - (render_normal * N_med).sum(dim=0)) * mask
        loss = loss_map.mean()
        return loss

class Pi3NormalLoss:
    def __init__(self, iter_from, iter_end):
        self.iter_from = iter_from
        self.iter_end = iter_end

    def is_active(self, iteration):
        return iteration >= self.iter_from and iteration <= self.iter_end

    def __call__(self, cam, render_pkg, iteration):
        assert "raw_normal" in render_pkg, "Forgot to set `output_normal=True` when calling render?"

        if not self.is_active(iteration):
            return 0
        gt = cam.normal              # [3,H,W]
        conf = cam.conf              # [1,H,W]
        pred = render_pkg["raw_normal"]  # [3,H,W]
        device = pred.device

        if gt is None:
            return torch.tensor(0.0, device=device)
        
        Hp, Wp = pred.shape[-2], pred.shape[-1]
        Hg, Wg = gt.shape[-2], gt.shape[-1]
        if (Hg, Wg) != (Hp, Wp):
            gt = F.interpolate(gt.unsqueeze(0), size=(Hp, Wp),
                               mode='bilinear', align_corners=False).squeeze(0)
            if conf is not None:
                conf = F.interpolate(conf.unsqueeze(0), size=(Hp, Wp),
                                   mode='bilinear', align_corners=False).squeeze(0)
        
        

        gt = gt.to(device)

        gt = F.normalize(gt, dim=0, eps=1e-8)
        pred = F.normalize(pred, dim=0, eps=1e-8)

        valid = (gt.abs().sum(dim=0) > 0)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device)

        cos = (pred * gt).sum(dim=0).clamp(-1.0, 1.0)  # [H,W]
        err = 1.0 - cos  # [H,W]
        w = conf.to(device).squeeze(0) if conf is not None else torch.ones_like(err, device=device)
        loss = (w[valid] * err[valid]).sum() / w[valid].sum()
        return loss
        
'''
import torch
import torch.nn.functional as F

class Pi3NormalLoss:
    def __init__(self, iter_from, iter_end, neighbor='4', gamma=2.0, tau=0.0):
        """
        neighbor: '4' or '8' (이웃 개수)
        gamma   : 가중치 날카로움(클수록 outlier 더 강하게 downweight)
        tau     : 코사인 기준 오프셋(>0로 주면 더 엄격)
        """
        self.iter_from = iter_from
        self.iter_end = iter_end
        self.neighbor = neighbor
        self.gamma = gamma
        self.tau = tau

    def is_active(self, iteration):
        return iteration >= self.iter_from and iteration <= self.iter_end

    @staticmethod
    def _shift(t, dy, dx):
        # t: [B=1, C=3, H, W]
        t = F.pad(t, (max(dx,0), max(-dx,0), max(dy,0), max(-dy,0)), mode='replicate')
        H, W = t.shape[-2:]
        return t[..., max(-dy,0):H-max(dy,0), max(-dx,0):W-max(dx,0)]

    def __call__(self, cam, render_pkg, iteration):
        assert "raw_normal" in render_pkg, "Forgot to set `output_normal=True` when calling render?"

        if not self.is_active(iteration):
            return torch.tensor(0.0, device=render_pkg["raw_normal"].device)

        gt   = cam.normal              # [3,H,W]
        pred = render_pkg["raw_normal"]  # [3,H,W]
        device = pred.device

        if gt is None:
            return torch.tensor(0.0, device=device)

        # resize gt -> pred size if needed
        Hp, Wp = pred.shape[-2], pred.shape[-1]
        Hg, Wg = gt.shape[-2], gt.shape[-1]
        if (Hg, Wg) != (Hp, Wp):
            gt = F.interpolate(gt.unsqueeze(0), size=(Hp, Wp),
                               mode='bilinear', align_corners=False).squeeze(0)

        gt   = gt.to(device)
        gt   = F.normalize(gt, dim=0, eps=1e-8)
        pred = F.normalize(pred, dim=0, eps=1e-8)

        # valid mask: gt가 (0,0,0)이 아닌 곳
        valid = (gt.abs().sum(dim=0) > 0)  # [H,W]
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device)

        # ----- 이웃 일관성 기반 weight 계산 (gt만 사용: 예측 값 의존성 제거) -----
        gtb = gt.unsqueeze(0)   # [1,3,H,W]
        nbrs = [(0,1), (0,-1), (1,0), (-1,0)]
        if self.neighbor == '8':
            nbrs += [(1,1), (1,-1), (-1,1), (-1,-1)]

        cos_list = []
        for (dy,dx) in nbrs:
            gtn = self._shift(gtb, dy, dx)                   # [1,3,H,W]
            gtn = F.normalize(gtn, dim=1, eps=1e-8)
            cosn = (gtb * gtn).sum(dim=1).clamp(-1.0, 1.0)   # [1,H,W]
            cos_list.append(cosn)

        cos_stack = torch.stack(cos_list, dim=0)              # [K,1,H,W]
        mean_cos  = cos_stack.mean(dim=0).squeeze(0)          # [H,W], -1~1

        # 가중치: (mean_cos - tau) -> [0,1]로 클램프 후 감마 적용
        w = (mean_cos - self.tau).clamp(min=0.0, max=1.0)
        w = w.pow(self.gamma)                                  # [H,W]
        # 경계/invalid 억제
        w = w * valid

        # weight는 학습 안정성을 위해 detach 권장 (원하면 빼도 됨)
        w = w.detach()

        # ----- 최종 손실: (1 - cos(pred, gt)) * w -----
        cos_pg = (pred * gt).sum(dim=0).clamp(-1.0, 1.0)       # [H,W]
        base = 1.0 - cos_pg                                    # [H,W]

        # 유효 + 가중 적용
        weighted = base * w
        # 평균을 공정하게 하려면 정규화(가중치 총합)로 나누기
        denom = w.sum().clamp_min(1.0)
        loss = weighted.sum() / denom
        return loss
'''