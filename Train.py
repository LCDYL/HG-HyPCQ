# -*- coding: UTF-8 -*-

import random
import warnings
import copy
import shutil

from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.data import WeightedRandomSampler
from tqdm import *

from Model import *
from MyDataset import *
from config import Config as Config

# 忽略所有UserWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

# ====== EEG 根目录和数据集名称 ======
EEG_ROOT = r'../1-simple_data_process/'
ALL_DOMAINS = ['2778', '3940', '4584']
# =====================================================

config = Config()

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def result_calculate(y_true, y_pred, y_prob):
    """
    y_true: list[int] 0/1
    y_pred: list[int] 0/1
    y_prob: list[float] = P(y==1)  (正类概率)
    return: dict
    """
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    ys = np.asarray(y_prob, dtype=np.float64)

    out = {}
    out["Acc"] = float(accuracy_score(yt, yp))
    out["BA"] = float(balanced_accuracy_score(yt, yp))
    out["F1"] = float(f1_score(yt, yp))

    # AUROC / AUPR 需要同时存在正负类，否则 sklearn 会报错；这里按“尽量少健壮性”处理：直接 try
    try:
        out["AUROC"] = float(roc_auc_score(yt, ys))
    except Exception:
        out["AUROC"] = float("nan")
    try:
        out["AUPR"] = float(average_precision_score(yt, ys))
    except Exception:
        out["AUPR"] = float("nan")

    return out

def extract_emb(backbone_model: MySuperEEG, x: torch.Tensor) -> torch.Tensor:
    # x: [B,C,T]
    E = backbone_model.encoder(x)  # [B,C,Np,D]
    E_prime = backbone_model.eeg_position_coder(E)  # [B,C,Np,D]
    emb = E_prime.mean(dim=(1, 2))  # [B,D]
    return emb


if __name__ == '__main__':
    # ========= 解包所有用到的配置 =========
    Code_State           = config.State
    batch_size           = config.batch_size
    test_rate            = config.test_rate
    GPU                  = config.GPU
    epochs               = config.epochs
    save_every           = config.save_every
    learning_rate        = config.lr
    seed                 = config.seed
    seed_all(seed)

    emb_dim              = config.emb_dim
    pq_M                 = config.pq_M
    pq_K                 = config.pq_K
    pq_softmax_temp      = config.pq_softmax_temp
    pq_quant_method      = config.pq_quant_method
    pq_init_neg_curvs    = config.pq_init_neg_curvs
    pq_clip_r            = config.pq_clip_r
    pq_use_alpha         = config.pq_use_alpha

    Spatial_Area_x1      = config.Spatial_Area_x1  # 最小超边
    Spatial_Area_x2      = config.Spatial_Area_x2  # 最大超边
    Time_Area_n          = config.Time_Area_n  # 时间范围大小

    margin_M             = config.margin_M  # 层次聚类损失里'类间分离'项的间隔系数
    w_min                = config.w_min  # 各层损失权重的最小值
    w_max                = config.w_max  # 各层损失权重的最大值

    mask_ratio           = config.mask_ratio
    recon_weight         = config.recon_weight
    cluster_weight       = config.cluster_weight

    sfrq = config.sfrq
    time_window_length   = config.time_window_length
    time_window_overlap  = config.time_window_overlap
    patch_length         = config.patch_length
    Target_domain_Name   = config.target_domain

    print('State:', Code_State)

    # ========= 设备 =========
    if torch.cuda.is_available():
        device = torch.device('cuda:' + GPU)
        print('Using GPU:', device)
    else:
        device = torch.device('cpu')
        print('Using CPU')

    win_samples = time_window_length * sfrq

    # ========= 三分支：Train / Finetune / Test =========
    os.makedirs('./Result_Model', exist_ok=True)

    if Code_State == 'Train':
        print("==== 预训练（自监督：重建 + 层次聚类） ====")

        # -------- 1) 找到目标域并排除，只保留两个源域 --------
        Source_domain_Names = [d for d in ALL_DOMAINS if d != Target_domain_Name]
        assert len(Source_domain_Names) == 2, "当前实现按“两个源域”写"

        # -------- 2) 初始化两个源域 dataset（用你新的 MyDataset.py）--------
        name2cls = {"2778": D2778S, "3940": D3940S, "4584": D4584S}

        ds_a_name, ds_b_name = Source_domain_Names[0], Source_domain_Names[1]
        ds_a = name2cls[ds_a_name](root_dir=os.path.join(EEG_ROOT, ds_a_name))
        ds_b = name2cls[ds_b_name](root_dir=os.path.join(EEG_ROOT, ds_b_name))

        train_full = ConcatDataset([ds_a, ds_b])

        # -------- 3) 用 WeightedRandomSampler 做“域大小不平衡”处理 --------
        len_a, len_b = len(ds_a), len(ds_b)
        w_a = 1.0 / float(len_a)
        w_b = 1.0 / float(len_b)

        weights = torch.empty(len_a + len_b, dtype=torch.double)
        weights[:len_a] = w_a
        weights[len_a:] = w_b

        # 每个 epoch 抽多少个样本？
        # 用 len(train_full) 是最简单的定义；在该设置下，期望两个域各抽一半
        num_samples = len(train_full)

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=num_samples,
            replacement=True,  # 关键：允许小域被重复采样，才能“补齐”不平衡
        )

        train_loader = DataLoader(
            train_full,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
        )

        # -------- 4) 模型初始化--------
        model = MySuperEEG(
            win=win_samples,
            patch_length=patch_length,
            sfrq=sfrq,
            Spatial_Area_x1=Spatial_Area_x1, Spatial_Area_x2=Spatial_Area_x2, Time_Area_n=Time_Area_n,
            enc_out_dim=emb_dim,
            pq_M=pq_M, pq_K=pq_K,
            pq_softmax_temp=pq_softmax_temp, pq_quant_method=pq_quant_method,
            pq_init_neg_curvs=pq_init_neg_curvs, pq_clip_r=pq_clip_r, pq_use_alpha=pq_use_alpha,
            mask_ratio=mask_ratio,
            recon_weight=recon_weight, cluster_weight=cluster_weight,
            w_min=w_min, w_max=w_max,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

        ckpt_final = './Result_Model/EEG_pretrain_final.pth'
        ckpt_every = './Result_Model/EEG_pretrain_epoch{:03d}.pth'

        avg_loss = 0.0
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss, n = 0.0, 0
            for x, y in tqdm(train_loader):
                # x: [B, 29, 5000]
                x = x.to(device).float()
                optimizer.zero_grad(set_to_none=True)
                _, loss = model(x)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                running_loss += float(loss.item())
                n += 1

            avg_loss = running_loss / max(1, n)
            print(f"[Pretrain] epoch {epoch} | loss={avg_loss:.6f}")

            if epoch % save_every == 0:
                torch.save({
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'train_loss': avg_loss,
                    'cfg': dict(
                        src_domains=Source_domain_Names,
                        batch_size=batch_size,
                        sampler="WeightedRandomSampler(domain-balanced)",
                        num_samples=num_samples,
                    )
                }, ckpt_every.format(epoch))
                print(f"[Pretrain] snapshot -> {ckpt_every.format(epoch)}")

        torch.save({
            'model': model.state_dict(),
            'epoch': epochs,
            'train_loss': avg_loss,
            'cfg': dict(
                src_domains=Source_domain_Names,
                batch_size=batch_size,
                sampler="WeightedRandomSampler(domain-balanced)",
                num_samples=num_samples,
            )
        }, ckpt_final)
        print(f"*** final pretrain saved: {ckpt_final}")



    elif Code_State == 'Finetune':
        print("==== 微调（主干+分类头一起训练；5折；加权交叉熵；选模BA；保存为文件夹：backbone+clf） ====")

        # -------- 1) 源域：加载两个源域数据（同 Train 分支逻辑）--------
        Source_domain_Names = [d for d in ALL_DOMAINS if d != Target_domain_Name]
        assert len(Source_domain_Names) == 2, "当前微调实现按“两个源域”写"

        name2cls = {"2778": D2778S, "3940": D3940S, "4584": D4584S}

        ds_a_name, ds_b_name = Source_domain_Names[0], Source_domain_Names[1]
        ds_a = name2cls[ds_a_name](root_dir=os.path.join(EEG_ROOT, ds_a_name))
        ds_b = name2cls[ds_b_name](root_dir=os.path.join(EEG_ROOT, ds_b_name))
        train_full = ConcatDataset([ds_a, ds_b])

        # -------- 2) 取所有样本标签，用于 StratifiedKFold --------
        labels = []
        for i in range(len(train_full)):
            labels.append(int(train_full[i][1]))
        labels = np.asarray(labels, dtype=np.int64)
        indices = np.arange(len(train_full))

        # -------- 3) 载入预训练骨干（不冻结，后续一起训练）--------
        # pretrain_path = './Result_Model/EEG_pretrain_final.pth'
        pretrain_path = './Result_Model/EEG_pretrain_epoch240.pth'
        ckpt = torch.load(pretrain_path, map_location=device)

        backbone = MySuperEEG(
            win=win_samples,
            patch_length=patch_length,
            sfrq=sfrq,
            Spatial_Area_x1=Spatial_Area_x1, Spatial_Area_x2=Spatial_Area_x2, Time_Area_n=Time_Area_n,
            enc_out_dim=emb_dim,
            pq_M=pq_M, pq_K=pq_K,
            pq_softmax_temp=pq_softmax_temp, pq_quant_method=pq_quant_method,
            pq_init_neg_curvs=pq_init_neg_curvs, pq_clip_r=pq_clip_r, pq_use_alpha=pq_use_alpha,
            mask_ratio=mask_ratio,
            recon_weight=recon_weight, cluster_weight=cluster_weight,
            w_min=w_min, w_max=w_max,
        ).to(device)

        backbone.load_state_dict(ckpt['model'], strict=False)
        backbone.train()

        # -------- 4) 5折交叉验证设置 --------
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        exp_name = f'EEG_DG_tgt_{Target_domain_Name}'

        # “全局最佳模型”按 BA 选
        global_best_BA = -1.0
        global_best_metrics = None
        global_best_dir = None

        # 正类 index（默认 1 是 PD=1, HC=0）
        pos_index = 1

        backbone.load_state_dict(ckpt['model'], strict=False)
        init_backbone_state = copy.deepcopy(backbone.state_dict())
        for fi, (tr_idx, va_idx) in enumerate(skf.split(indices, labels), 1):
            print(f"\n===== Finetune Fold {fi}/5 | train={len(tr_idx)} val={len(va_idx)} =====")
            backbone.load_state_dict(init_backbone_state, strict=True)  # 或 strict=False
            backbone.train()

            # -------- 4.1) DataLoader--------
            tr_set = Subset(train_full, tr_idx.tolist())
            va_set = Subset(train_full, va_idx.tolist())

            tr_loader = DataLoader(
                tr_set,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0,
                pin_memory=True,
            )
            va_loader = DataLoader(
                va_set,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                pin_memory=True,
            )

            # -------- 4.2) 分类头 + 加权交叉熵 --------
            clf = Classification_Head(emb_dim).to(device)
            # 主干 + 分类头一起训练（一个 optimizer 管两部分参数）
            optimizer = torch.optim.Adam(
                list(backbone.parameters()) + list(clf.parameters()),
                lr=learning_rate
            )
            # 类权重：按训练折统计，反比例权重（少类更大）
            y_tr = labels[tr_idx]
            n_per_class = np.bincount(y_tr, minlength=2).astype(np.float32)  # 二分类
            class_weights_np = (n_per_class.sum() / (2.0 * np.maximum(n_per_class, 1.0))).astype(np.float32)
            class_weights = torch.tensor(class_weights_np, device=device, dtype=torch.float32)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
            # criterion = torch.nn.CrossEntropyLoss().to(device)
            # fold 内最佳（按 BA）
            fold_best_BA = -1.0
            fold_best_metrics = None
            fold_best_state = None  # 存 backbone+clf

            # -------- 4.3) 训练循环 --------
            for epoch in range(1, epochs + 1):
                backbone.train()
                clf.train()
                tr_loss_sum, tr_n = 0.0, 0
                for x, y in tr_loader:
                    x = x.to(device).float()  # [B,29,win]
                    y = y.to(device).long()  # [B]
                    emb = extract_emb(backbone, x)  # [B,D]
                    logits = clf(emb)  # [B,2]
                    loss = criterion(logits, y)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    tr_loss_sum += float(loss.item())
                    tr_n += 1

                tr_loss = tr_loss_sum / max(1, tr_n)

                # -------- 4.4) 验证：收集 y_true / y_pred / y_prob -> 统一算指标 --------
                backbone.eval()
                clf.eval()
                y_true_list, y_pred_list, y_prob_list = [], [], []

                with torch.no_grad():
                    for x, y in va_loader:
                        x = x.to(device).float()
                        y = y.to(device).long()

                        emb = extract_emb(backbone, x)  # [B,D]
                        logits = clf(emb)  # [B,2]
                        prob = F.softmax(logits, dim=1)[:, pos_index]  # [B]
                        pred = logits.argmax(dim=1)  # [B]

                        y_true_list.extend(y.detach().cpu().tolist())
                        y_pred_list.extend(pred.detach().cpu().tolist())
                        y_prob_list.extend(prob.detach().cpu().tolist())

                metrics = result_calculate(y_true_list, y_pred_list, y_prob_list)

                print(
                    f"[Fold {fi}] Epoch {epoch} | train_loss={tr_loss:.4f} | "
                    f"val_Acc={metrics['Acc']:.4f} | val_BA={metrics['BA']:.4f} | "
                    f"val_AUROC={metrics['AUROC']:.4f} | val_F1={metrics['F1']:.4f} | val_AUPR={metrics['AUPR']:.4f}"
                )

                # fold 内按 BA 选最佳：需要同时 snapshot backbone + clf
                if metrics["BA"] > fold_best_BA:
                    fold_best_BA = metrics["BA"]
                    fold_best_metrics = metrics
                    fold_best_state = {
                        "backbone": copy.deepcopy(backbone.state_dict()),
                        "clf": copy.deepcopy(clf.state_dict()),
                        "fold": fi,
                        "epoch": epoch,
                        "metrics": metrics,
                        "emb_dim": emb_dim,
                        "src_domains": Source_domain_Names,
                        "tgt_domain": Target_domain_Name,
                    }

            # -------- 4.5) Fold 结束：按 BA 与全局最优对比，决定是否保存（保存为文件夹：backbone+clf）--------
            # 你原本逻辑：Fold1直接保存；后续fold若更好覆盖
            if (fi == 1) or (fold_best_BA > global_best_BA):
                global_best_BA = fold_best_BA
                global_best_metrics = fold_best_metrics

                save_dir = f'./Result_Model/{exp_name}_finetune_fold{fi}_best'
                global_best_dir = save_dir

                # 覆盖式保存：先删目录再建（避免残留旧文件）
                if os.path.isdir(save_dir):
                    shutil.rmtree(save_dir)
                os.makedirs(save_dir, exist_ok=True)

                # 两个部分分别保存
                torch.save(fold_best_state, os.path.join(save_dir, "checkpoint.pth"))
                torch.save(fold_best_state["backbone"], os.path.join(save_dir, "backbone.pth"))
                torch.save(fold_best_state["clf"], os.path.join(save_dir, "clf.pth"))

                print(f"*** [Fold {fi}] New GLOBAL best by BA={global_best_BA:.4f} -> saved to folder: {save_dir}")
            else:
                print(
                    f"[Fold {fi}] best BA={fold_best_BA:.4f} (not better than global {global_best_BA:.4f}) -> not saved")

        # -------- 5) 打印“全局最佳模型”的五个指标 --------
        print("\n==== Finetune Finished ====")
        if global_best_metrics is None:
            print("No best model saved (unexpected).")
        else:
            print(f"Best model folder: {global_best_dir}")
            print(
                f"BEST Metrics | "
                f"Acc={global_best_metrics['Acc']:.4f} | "
                f"BA={global_best_metrics['BA']:.4f} | "
                f"AUROC={global_best_metrics['AUROC']:.4f} | "
                f"F1={global_best_metrics['F1']:.4f} | "
                f"AUPR={global_best_metrics['AUPR']:.4f}"
            )



    elif Code_State == 'Test':
        print("==== 测试（加载微调最佳模型；目标域测试；保存 metrics.txt + preds.csv） ====")

        # -------- 1) 构造目标域 dataset / dataloader --------
        # 注意：按你说明 DxxxS 中的 S 与源/目标无关，所以这里直接用 D2778S/D3940S/D4584S 映射
        name2cls = {"2778": D2778S, "3940": D3940S, "4584": D4584S}
        if Target_domain_Name not in name2cls:
            raise SystemExit(
                f"[Test] Target_domain_Name={Target_domain_Name} 不在 {list(name2cls.keys())} 中。\n"
                f"请在 Test 分支里补充 name2cls 映射。"
            )

        tgt_root = os.path.join(EEG_ROOT, Target_domain_Name)
        test_set = name2cls[Target_domain_Name](root_dir=tgt_root)

        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
        )

        print(f"[Test] domain={Target_domain_Name} | test_size={len(test_set)}")

        # -------- 2) 自动找到“微调保存的最佳模型文件夹”（按 checkpoint 里的 BA 最大）--------
        exp_name = f'EEG_DG_tgt_{Target_domain_Name}'

        import glob as _glob
        # pattern = f'./Result_Model/{exp_name}_finetune_fold*_best'
        pattern = f'./Result_Model/{exp_name}_finetune_fold1_best'
        cand_dirs = sorted([d for d in _glob.glob(pattern) if os.path.isdir(d)])

        if len(cand_dirs) == 0:
            raise SystemExit(
                f"[Test] 没找到任何微调模型文件夹：pattern={pattern}\n"
                f"请先运行 Finetune，或检查 Result_Model 路径。"
            )

        best_dir = None
        best_ba = -1.0
        best_ckpt = None

        for d in cand_dirs:
            ckpt_path = os.path.join(d, "checkpoint.pth")
            if not os.path.isfile(ckpt_path):
                continue
            try:
                tmp = torch.load(ckpt_path, map_location="cpu")
                ba = float(tmp.get("metrics", {}).get("BA", -1.0))
            except Exception:
                tmp, ba = None, -1.0

            if ba > best_ba:
                best_ba = ba
                best_dir = d
                best_ckpt = tmp

        if best_dir is None or best_ckpt is None:
            raise SystemExit(
                f"[Test] 找到了候选目录但无法读取有效 checkpoint.pth：{cand_dirs}\n"
                f"请检查保存是否完整。"
            )

        print(f"[Test] Auto-selected BEST folder: {best_dir} (BA={best_ba:.4f})")

        # -------- (可选) 如果你想手动指定某一折，使用下面两行替换上面的自动选择即可：--------
        # best_dir = f'./Result_Model/{exp_name}_finetune_fold3_best'
        # best_ckpt = torch.load(os.path.join(best_dir, "checkpoint.pth"), map_location="cpu")

        # -------- 3) 构造 backbone + clf 并加载权重 --------
        backbone = MySuperEEG(
            win=win_samples,
            patch_length=patch_length,
            sfrq=sfrq,
            Spatial_Area_x1=Spatial_Area_x1, Spatial_Area_x2=Spatial_Area_x2, Time_Area_n=Time_Area_n,
            enc_out_dim=emb_dim,
            pq_M=pq_M, pq_K=pq_K,
            pq_softmax_temp=pq_softmax_temp, pq_quant_method=pq_quant_method,
            pq_init_neg_curvs=pq_init_neg_curvs, pq_clip_r=pq_clip_r, pq_use_alpha=pq_use_alpha,
            mask_ratio=mask_ratio,
            recon_weight=recon_weight, cluster_weight=cluster_weight,
            w_min=w_min, w_max=w_max,
        ).to(device)

        clf = Classification_Head(emb_dim).to(device)

        backbone.load_state_dict(best_ckpt["backbone"], strict=True)
        clf.load_state_dict(best_ckpt["clf"], strict=True)

        backbone.eval()
        clf.eval()

        # 正类 index（默认 1 是 PD=1, HC=0）
        pos_index = 1

        # -------- 4) 跑测试：收集 y_true / y_pred / y_prob --------
        y_true_list, y_pred_list, y_prob_list = [], [], []

        with torch.no_grad():
            for x, y in tqdm(test_loader, desc="[Test]"):
                x = x.to(device).float()   # [B, C, T]
                y = y.to(device).long()    # [B]

                emb = extract_emb(backbone, x)       # [B, D]
                logits = clf(emb)                    # [B, 2]
                prob = F.softmax(logits, dim=1)[:, pos_index]  # [B]  (P(y==1))
                pred = logits.argmax(dim=1)          # [B]

                y_true_list.extend(y.detach().cpu().tolist())
                y_pred_list.extend(pred.detach().cpu().tolist())
                y_prob_list.extend(prob.detach().cpu().tolist())

        metrics = result_calculate(y_true_list, y_pred_list, y_prob_list)

        # -------- 5) 保存：./TestResult/<timestamp>/{metrics.txt, preds.csv} --------
        # 用到 time（你当前文件顶部没 import time，所以这里本地 import 一下，避免影响其他分支）
        import time as _time
        ts = _time.strftime("%Y%m%d_%H%M%S")

        out_dir = os.path.join("./TestResult", ts)
        os.makedirs(out_dir, exist_ok=True)

        # 5.1 保存 preds.csv
        import csv as _csv
        preds_csv = os.path.join(out_dir, "preds.csv")
        with open(preds_csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["y_true", "y_pred", "y_prob"])
            for yt, yp, yp_prob in zip(y_true_list, y_pred_list, y_prob_list):
                w.writerow([int(yt), int(yp), float(yp_prob)])

        # 5.2 保存 metrics.txt
        metrics_txt = os.path.join(out_dir, "metrics.txt")
        with open(metrics_txt, "w", encoding="utf-8") as f:
            f.write("==== Test Result ====\n")
            f.write(f"timestamp: {ts}\n")
            f.write(f"target_domain: {Target_domain_Name}\n")
            f.write(f"test_size: {len(test_set)}\n")
            f.write(f"loaded_model_dir: {best_dir}\n")
            f.write(f"selected_by_checkpoint_BA: {best_ba:.6f}\n\n")

            f.write("---- Metrics ----\n")
            f.write(f"Acc:   {metrics['Acc']:.6f}\n")
            f.write(f"BA:    {metrics['BA']:.6f}\n")
            f.write(f"AUROC: {metrics['AUROC']:.6f}\n")
            f.write(f"F1:    {metrics['F1']:.6f}\n")
            f.write(f"AUPR:  {metrics['AUPR']:.6f}\n")

        print("\n==== Test Finished ====")
        print(f"Saved folder: {out_dir}")
        print(f"  - {metrics_txt}")
        print(f"  - {preds_csv}")
        print(
            f"TEST Metrics | "
            f"Acc={metrics['Acc']:.4f} | "
            f"BA={metrics['BA']:.4f} | "
            f"AUROC={metrics['AUROC']:.4f} | "
            f"F1={metrics['F1']:.4f} | "
            f"AUPR={metrics['AUPR']:.4f}"
        )
