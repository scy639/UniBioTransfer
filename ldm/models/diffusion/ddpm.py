
from .misc_4ddpm import *
from lmk_util.lmk_extractor import lmkAll_2_lmkMain, get_lmkMain_indices

class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 learn_logvar=False,
                 logvar_init=0.,
                 u_cond_percent=0,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size 
        self.channels = channels
        self.u_cond_percent=u_cond_percent
        unet_config['params']['in_channels'] = 14 if CH14 else 9
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)


    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        assert 0
        print("[init_from_ckpt]")
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none') #-->
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        assert 0, 'This should not be called; subclasses override this method'
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        # metrics.csv entries like 'train/...' and 'val/...' originate here
        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)


    def shared_step(self, batch):
        assert 0

            
    def set_task(self, batch):
        task = batch['task'][0].item()
        printC('task',f"{task=}")
        global_.task = task
        assert all(batch['task'] == task), batch['task']
        self.task = task
        if 1:
            if (not USE_pts) or task==1:    self.Landmark_cond=False
            else:    self.Landmark_cond=True
        if 1:
            if task in (0,2,3,):
                self.Landmarks_weight=0.05
            else:
                self.Landmarks_weight=0
            self.STACK_feat=True
        return task
    def unset_task(self):
        global_.task = None
        global_.lmk_ = None
        del self.task
    def training_step(self, batch, batch_idx):
        task = batch['task'][0].item()
        opt = self.optimizers()
        
        if not self.Reconstruct_initial:# only MSE loss(orig diffusion). -> shared_step -> forward -> p_losses
            loss, loss_dict = self.shared_step(batch)       # original
        else: # added Multistep (DDIM) loss -> shared_step_face -> forward_face -> p_losses_face
            loss, loss_dict = self.shared_step_face(batch) # changed by sanoojan : to add ID loss after reconstructing through DDIM

        step_or_accumulate = ( task==3 or TP_enable)
        _ctx = nullcontext
        if not step_or_accumulate and not TP_enable:
            _ctx = self.trainer.model.no_sync # https://github.com/Lightning-AI/pytorch-lightning/discussions/10792
        with _ctx(): # https://zhuanlan.zhihu.com/p/250471767
            self.manual_backward(loss)
        
        if (REFNET.ENABLE and REFNET.task2layerNum[task]>0):
            self.model.bank.clear()
        self.unset_task()
        
        
        total_step = len(self.trainer.train_dataloader)
        if step_or_accumulate:
            # Average grads of shared params across ranks (TaskParallel)
            if dist.is_available() and dist.is_initialized():
                ws = dist.get_world_size()
                shared_sync_cnt = 0; task_skip_cnt = 0
                for name, p in self.named_parameters():
                    need_sync, is_task_specific_skip = tp_param_need_sync(name, p)
                    if not need_sync:
                        if is_task_specific_skip:
                            task_skip_cnt += 1
                        continue
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # ensure collective call sequence remains consistent
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad.div_(ws)
                    shared_sync_cnt += 1
                if gate_('[TP] shared sync counts'):
                    print(f"synced={shared_sync_cnt} skipped(task)={task_skip_cnt}")
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            opt.step()
            opt.zero_grad()
            if self.use_scheduler: # handle LR schedulers
                sch = self.lr_schedulers()
                if isinstance(sch, list) and len(sch) > 0: # schedulers expressed as a list
                    for scheduler_config in sch:
                        if isinstance(scheduler_config, dict) and 'scheduler' in scheduler_config:
                            scheduler_config['scheduler'].step()
                        else:
                            scheduler_config.step()
                elif hasattr(sch, 'step'):
                    sch.step()
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss
    # manual optimization calls backward in training_step already, so this is skipped here
    # def backward(

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.unset_task()
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)




class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.automatic_optimization = False # disable automatic optimization to manage parameter updates manually
     
        
        # self.learnable_vector = nn.Parameter(torch.randn((1,1,768)), requires_grad=True)
        # breakpoint()
        
         
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        
        #check if other_params is present in cond_stage_config
        if hasattr(cond_stage_config, 'other_params'):
        
            self.clip_weight=cond_stage_config.other_params.clip_weight
            # those three weights: 0 skips module init, >0 enables it and acts as weight when !STACK_feat
            if set(TASKS) & {0,2,3}:   self.ID_weight = 10.0
            else:   self.ID_weight = 0
            if (not USE_pts) and TASKS==(1,):    self.Landmark_cond=False
            else:    self.Landmark_cond=True
            self.Landmarks_weight=0.05
            if hasattr(cond_stage_config.other_params, 'Additional_config'):
                self.Reconstruct_initial=cond_stage_config.other_params.Additional_config.Reconstruct_initial
                self.Reconstruct_DDIM_steps=cond_stage_config.other_params.Additional_config.Reconstruct_DDIM_steps
                self.sampler=DDIMSampler(self)
                if hasattr(cond_stage_config.other_params, 'multi_scale_ID'):
                    self.multi_scale_ID=cond_stage_config.other_params.multi_scale_ID   # True has an issue
                else:
                    self.multi_scale_ID=True  #this has an issue obtaining earlier layer from ID
                if hasattr(cond_stage_config.other_params, 'normalize'):
                    self.normalize=cond_stage_config.other_params.normalize  # normalizes the combintaion of ID and LPIPS loss
                else:
                    self.normalize=False
                if 1:
                    self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
                if hasattr(cond_stage_config.other_params, 'partial_training'):
                    self.partial_training=cond_stage_config.other_params.partial_training
                    self.trainable_keys=cond_stage_config.other_params.trainable_keys
                else:
                    self.partial_training=False
                if hasattr(cond_stage_config.other_params.Additional_config, 'Same_image_reconstruct'):
                    self.Same_image_reconstruct=cond_stage_config.other_params.Additional_config.Same_image_reconstruct
                else:
                    self.Same_image_reconstruct=False
                if hasattr(cond_stage_config.other_params.Additional_config, 'Target_CLIP_feat'):
                    self.Target_CLIP_feat=cond_stage_config.other_params.Additional_config.Target_CLIP_feat
                else:
                    self.Target_CLIP_feat=False
                if hasattr(cond_stage_config.other_params.Additional_config, 'Source_CLIP_feat'):
                    self.Source_CLIP_feat=cond_stage_config.other_params.Additional_config.Source_CLIP_feat
                else:
                    self.Source_CLIP_feat=False
                if hasattr(cond_stage_config.other_params.Additional_config, 'use_3dmm'):  
                    self.use_3dmm=cond_stage_config.other_params.Additional_config.use_3dmm
                else:
                    self.use_3dmm=False
                    
            else:
                self.Reconstruct_initial=False
                self.Reconstruct_DDIM_steps=0
                
            self.update_weight=False  
                
        else:
            assert 0
        if 1:
            self.learnable_vector = nn.ParameterList([
                nn.Parameter(torch.randn((1,259,768)), requires_grad=True),
                nn.Parameter(torch.randn((1,257,768)), requires_grad=True),
                nn.Parameter(torch.randn((1,259,768)), requires_grad=True),
                nn.Parameter(torch.randn((1,259,768)), requires_grad=True),
            ])
        if self.ID_weight>0:
            if self.multi_scale_ID:
                self.ID_proj_out=nn.Linear(200704, 768)
            else:
                self.ID_proj_out=nn.Linear(512, 768) # yes
            self.instantiate_IDLoss(cond_stage_config)
            
        if self.Landmark_cond:
            if USE_pts:
                self.ptsM_Generator = LandmarkExtractor(include_visualizer=True,img_256_mode=False)
            else:
                self.detector = dlib.get_frontal_face_detector()
                self.predictor = dlib.shape_predictor("Other_dependencies/DLIB_landmark_det/shape_predictor_68_face_landmarks.dat")
                        
            if self.Landmarks_weight>0:
                self.landmark_proj_out=nn.Linear(NUM_pts*2, 768)
        self.total_steps_in_epoch=0 # will be calculated inside training_step. Not known for now
        if 1:
            assert cond_stage_config.target=="ldm.modules.encoders.modules.FrozenCLIPEmbedder" and self.Source_CLIP_feat and self.Target_CLIP_feat
            self.USE_proj_out_source = 1
            if set(TASKS) & {0,}:
                self.proj_out_source__face=nn.Linear(768, 768)
            if set(TASKS) & {1,}:
                self.proj_out_source__hair=nn.Linear(768, 768)
            if set(TASKS) & {2,3,}:
                self.proj_out_source__head=nn.Linear(768, 768)
            if 0:    # dummy, just for compa
                self.proj_out_target=nn.Linear(768, 768)
                self.proj_out=nn.Identity()
        
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        
        
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None  

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def get_lmk_for_router(self, batch: dict, x_tensor: torch.Tensor):
        """
        Prepare global_.lmk_ (BS, L, 2) normalized to [0,1] for gating/router.
        - Prefer cached Mediapipe landmarks if present in batch
        - Convert 468/478 to main landmarks with face oval using get_lmkMain_indices(True)
        - Fallback to zeros if not available
        """
        b, _, H, W = x_tensor.shape
        if READ_mediapipe_result_from_cache and ('mediapipe_lmkAll' in batch):
            data_all = batch['mediapipe_lmkAll']  # tensor or ndarray
            if isinstance(data_all, torch.Tensor):
                lmks_all = data_all.to(x_tensor.device).to(x_tensor.dtype)
            else:
                lmks_all = torch.from_numpy(data_all).to(x_tensor.device).to(x_tensor.dtype)
            # map to main indices with face oval (cached tensor indices on device)
            idxs = getattr(global_, 'lmk_main_idx_tensor', None)
            if (idxs is None) or (idxs.device != x_tensor.device):
                idx_list = get_lmkMain_indices(include_face_oval=True)
                idxs = torch.as_tensor(list(idx_list), dtype=torch.long, device=x_tensor.device)
                global_.lmk_main_idx_tensor = idxs
            lmk = torch.index_select(lmks_all, dim=1, index=idxs)
            # normalize by current spatial size
            if lmk.numel() > 0:
                # print(f"0 {lmk[:,:5]=}")
                lmk[..., 0] = lmk[..., 0] / float(W)
                lmk[..., 1] = lmk[..., 1] / float(H)
                # print(f"1 {lmk[:,:5]=}")
        else:
            assert 0
            lmk = torch.zeros((b, 0, 2), device=x_tensor.device, dtype=x_tensor.dtype)
        return lmk

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert 0

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_IDLoss(self, config):
        # Need to modify @sanoojan
        # if not self.cond_stage_trainable:
        model = IDLoss(config,multiscale=self.multi_scale_ID)
        self.face_ID_model = model.eval()
        self.face_ID_model.train = disabled_train
        for param in self.face_ID_model.parameters():
            param.requires_grad = False
            
    
    
    def instantiate_cond_stage(self, config):
        if 1:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model: FrozenCLIPEmbedder = instantiate_from_config(config) #ldm.modules.encoders.modules.FrozenCLIPEmbedder
            if 0 in TASKS:
                self.encoder_clip_face :FrozenCLIPEmbedder = model
            if 1 in TASKS:
                self.encoder_clip_hair :FrozenCLIPEmbedder = copy.deepcopy(model)
                del self.encoder_clip_hair.model
                del self.encoder_clip_hair.tokenizer
            if set(TASKS) & {2,}:
                self.encoder_clip_head_t2 :FrozenCLIPEmbedder = copy.deepcopy(model)
                del self.encoder_clip_head_t2.model
                del self.encoder_clip_head_t2.tokenizer
            if set(TASKS) & {3,}:
                self.encoder_clip_head_t3 :FrozenCLIPEmbedder = copy.deepcopy(model)
                del self.encoder_clip_head_t3.model
                del self.encoder_clip_head_t3.tokenizer


    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                            force_not_quantize=force_no_decoder_quantization))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    
    def get_learned_conditioning(self, c):
        raise Exception
    def conditioning_with_feat(self,x,landmarks=None,enInputs:dict=None):
        if gate_('vis LatentDiffusion.conditioning_with_feat'):
            debug_dir = Path(f"4debug/conditioning_with_feat/{ID}"); debug_dir.mkdir(parents=0, exist_ok=True)
            all_images = [ ('x', x), ]
            for _name, _enInput in enInputs.items():
                all_images.append((_name, _enInput))
            vis_tensors_A(all_images, debug_dir / f"all-{str_t_pid()}.jpg", vis_batch_size= min(5, landmarks.shape[0]) )
        del x # (x is GT during training, ref_imgs during inference)
        task = self.task
        ID_weight = self.ID_weight
        Landmarks_weight = self.Landmarks_weight
        if self.task==0:
            face_clip_weight = self.clip_weight
        elif self.task==1:
            hair_clip_weight = self.clip_weight
        elif self.task==2:
            head_clip_weight = self.clip_weight
        elif self.task==3:
            head_clip_weight = self.clip_weight
        if 1:
            cs = [] # conditionings
            ws = [] # weights corresponding one-to-one with cs
            def encode_face_ID():
                _c = enInputs['face_ID-in']
                _c=self.face_ID_model.extract_feats(_c)[0]
                _c = self.ID_proj_out(_c) #-->c:[4,768]
                _c = _c.unsqueeze(1) #-->c:[4,1,768]
                if self.normalize: #normalize c2
                    _c = _c*norm_coeff/F.normalize(_c, p=2, dim=2)
                cs.append(_c);  ws.append(ID_weight)
            def encode_face_clip(_z=None):# _z: result of ViT forward pass
                if _z is None:
                    _c = enInputs['face-clip-in']
                    _c = self.encoder_clip_face.encode(_c)  #b,3,224,224 --> b,1,768
                else:
                    assert 0
                    _c = self.encoder_clip_face.encode_B(_z)
                if hasattr(self,'USE_proj_out_source') and self.USE_proj_out_source:
                    _c = self.proj_out_source__face(_c)
                cs.append(_c);  ws.append(face_clip_weight)
            def encode_hair_clip(_z=None):
                if _z is None:
                    _c = enInputs['hair-clip-in']
                    _c = self.encoder_clip_hair.encode(_c)  #b,3,224,224 --> b,1,768
                else:
                    _c = self.encoder_clip_hair.encode_B(_z)
                if hasattr(self,'USE_proj_out_source') and self.USE_proj_out_source:
                    _c = self.proj_out_source__hair(_c)
                printC("hair _c.shape:",f"{_c.shape}")
                cs.append(_c);  ws.append(hair_clip_weight)
            def encode_head_clip(_z=None):
                if global_.task == 2:
                    encoder_clip_head = self.encoder_clip_head_t2
                elif global_.task == 3:
                    encoder_clip_head = self.encoder_clip_head_t3
                else:
                    raise ValueError(f"Task {global_.task} does not have encoder_clip_head")
                if _z is None:
                    _c = enInputs['head-clip-in']
                    _c = encoder_clip_head.encode(_c)  #b,3,224,224 --> b,1,768
                else:
                    _c = encoder_clip_head.encode_B(_z)
                if hasattr(self,'USE_proj_out_source') and self.USE_proj_out_source:
                    _c = self.proj_out_source__head(_c)
                printC("head _c.shape:",f"{_c.shape}")
                cs.append(_c);  ws.append(head_clip_weight)
            if task==0:
                encode_face_ID()
                encode_face_clip()
            elif task==1:
                _z = enInputs['hair-clip-in']
                _z = self.encoder_clip_face.forward_vit(_z)
                encode_hair_clip(_z)
            elif task==2:
                encode_face_ID()
                _z = enInputs['head-clip-in']
                _z = self.encoder_clip_face.forward_vit(_z)
                encode_head_clip(_z)
            elif task==3:
                encode_face_ID()
                _z = enInputs['head-clip-in']
                _z = self.encoder_clip_face.forward_vit(_z)
                encode_head_clip(_z)
        c=0
            
        if Landmarks_weight > 0:
            landmarks=landmarks.unsqueeze(1) if len(landmarks.shape)!=3 else landmarks
            cs.append(landmarks);  ws.append(Landmarks_weight)
        if self.STACK_feat: # _Cc
            # stack all features
            conc=torch.cat(cs, dim=-2)
            c = conc
        else:
            total_weight = sum(ws)
            weighted_sum = sum(c * w for c, w in zip(cs, ws))
            c = weighted_sum / total_weight if total_weight > 0 else 0
        printC("[conditioning_with_feat return]",f"{custom_repr_v3(c)}")
        # assert c.shape[1]==NUM_token, c.shape
        return c
    

    def get_landmarks(self,x, batch:dict):
        
        if (self.Landmark_cond) and x is not None:
            # pass
            # # Detect faces in an image
            #convert to 8bit image
            x=255.0*un_norm(x).permute(0,2,3,1).cpu().numpy()
            x=x.astype(np.uint8) # B,512,512,3
            Landmarks_all=[]    
            if USE_pts:
                l_lmkAll=[]
                if READ_mediapipe_result_from_cache:
                    _l_lmkAll :np.ndarray = batch['mediapipe_lmkAll'].cpu().numpy()
            bs = len(x)
            for i in range(len(x)):
                if USE_pts:
                    if READ_mediapipe_result_from_cache:
                        lmkAll :np.ndarray = _l_lmkAll[i]
                    else:
                        lmkAll :np.ndarray = self.ptsM_Generator.extract_single(x[i], only_main_lmk=False)
                    if lmkAll is None: lmkAll = np.zeros((478,2))
                    l_lmkAll.append(lmkAll)
                    lm = lmkAll_2_lmkMain(lmkAll) # NUM_pts,2
                lm = lm.reshape(1, NUM_pts*2) # num of points * 2 coordinates
                Landmarks_all.append(lm)
                if 0:
                    from util_vis import visualize_landmarks
                    starter_stem = Path(sys.argv[0]).stem
                    path_vis_lmk = f'4debug/vis_lmk/{starter_stem}-{i}.png'
                    visualize_landmarks(x[i], lm[0], path_vis_lmk)
                    print(f"{path_vis_lmk=}")
        Landmarks_all=np.concatenate(Landmarks_all,axis=0)
        pts68 = Landmarks_all.reshape(bs, NUM_pts, 2, )
        if self.Landmarks_weight>0:
            Landmarks_all=torch.tensor(Landmarks_all).float().to(self.device)
            if self.Landmark_cond == False:
                return Landmarks_all
            with torch.enable_grad():
                Landmarks_all=self.landmark_proj_out(Landmarks_all)
            # normalize Landmarks_all
        
        lmk_aux={}
        if USE_pts: lmk_aux['l_lmkAll'] = l_lmkAll
        return Landmarks_all,pts68,lmk_aux

    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(weighting, self.split_input_params["clip_min_weight"],
                               self.split_input_params["clip_max_weight"], )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting,
                                     self.split_input_params["clip_min_tie_weight"],
                                     self.split_input_params["clip_max_tie_weight"])

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                                dilation=1, padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                                dilation=1, padding=0,
                                stride=(stride[0] // df, stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

    # returned x is the concatenated multi-channel tensor (mask, ref, lmk, ...); e.g. "x_start[:,8,:,:]" extracts the mask
    @torch.no_grad()
    def get_input_(self, batch, k, return_first_stage_outputs=False,
        cond_key=None, bs=None,
        get_referenceZ=False, # reference image latent tensor, dims B,4,64,64
    ):
        if k == "inpaint": # yes
            x = batch['GT']
            mask = batch['inpaint_mask'].clone() # b,1,512,512
            inpaint = batch['inpaint_image'].clone() # .clone so that batch['inpaint_image'] remains the original image without landmarks
            # reference = batch['ref_imgs']
            reference = None
        else:
            assert 0
        if len(x.shape) == 3:
            assert 0
            x = x[..., None]
        if 1:
            enInputs = batch['enInputs'] # encoder inputs (each self.encoder receives these raw tensors without preprocessing)
            for k,v in enInputs.items():
                enInputs[k] = v.to(memory_format=torch.contiguous_format).float()
        #--------------------------------------------------------------------------------
        ref_imgs_4unet = batch.get('ref_imgs_4unet', None) if get_referenceZ else None
        
        
        #x : Original Image
        #inpaint : Masked original image
        #mask: mask
        #reference: Transformed(Masked(original image))
        if bs is not None:
            assert 0
        x = x.to(self.device)
        
        global_.lmk_ = self.get_lmk_for_router(batch, x) # for router/gate
        if self.Landmark_cond: 
            landmarks, pts68, lmk_aux=self.get_landmarks(x,batch)
        else:
            landmarks=None
        
        if self.task in (0,2,3,) and USE_pts:
            mask_np = mask.detach().cpu().numpy()
            if 1:
                #convert to 8bit image
                x_unnorm=255.0*un_norm(x).permute(0,2,3,1).cpu().numpy()
                x_unnorm=x_unnorm.astype(np.uint8) # B,512,512,3
            
            batch_size = x.shape[0]
            
            VIS_pts= 0
            
            for b in range(batch_size):
                lmkAll = lmk_aux['l_lmkAll'][b]
                inpaint[b] = torch.Tensor(self.ptsM_Generator.visualizer.visualize_landmarks(inpaint[b].permute(1,2,0).detach().cpu().numpy(), lmkAll, ) ).permute(2,0,1)
                del lmkAll
            
        if self.training   and  gate_('vis LatentDiffusion.get_input'):
            debug_dir = Path(f"4debug/LatentDiffusion.get_input/{ID}"); debug_dir.mkdir(parents=0, exist_ok=True)
            vis_batch_size = min(5, x.shape[0])  # Show at most 4 samples
            all_images = [ ('x', x), ('inpaint', inpaint), ('mask', mask), ('reference', reference), ('ref_imgs_4unet', ref_imgs_4unet) ]
            for _name, _enInput in enInputs.items():
                all_images.append((_name, _enInput))
            all_path = debug_dir / f"all--after-pts-{str_t_pid()}.jpg"
            vis_tensors_A(all_images, all_path, vis_batch_size)
        
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        encoder_posterior_inpaint = self.encode_first_stage(inpaint)
        z_inpaint = self.get_first_stage_encoding(encoder_posterior_inpaint).detach()
        # tgt/ref_mask_64
        mask_resize = Resize([z.shape[-1],z.shape[-1]])(mask)
        ref_mask_64 = Resize([z.shape[-1],z.shape[-1]])(batch['ref_mask_512']) if 'ref_mask_512' in batch else None
        # z9 & z_ref
        if not CH14:
            z_new = torch.cat((z,z_inpaint,mask_resize),dim=1) # shape:[4,9,64,64] 9:4+4+1
        if get_referenceZ:
            encoder_posterior_ref = self.encode_first_stage(ref_imgs_4unet)
            z_ref = self.get_first_stage_encoding(encoder_posterior_ref).detach() # shape:[4,4,64,64]
        else:
            z_ref = None
        if  CH14:
            z_new = torch.cat((z,z_inpaint,mask_resize, z_ref,ref_mask_64),dim=1)
        assert z.shape[1:]==(4,64,64,)
        if  gate_(f'vis LatentDiffusion.get_input-before_return {self.training}'):
            debug_dir = Path(f"4debug/LatentDiffusion.get_input-before_return/{ID}"); debug_dir.mkdir(parents=0, exist_ok=True)
            vis_batch_size = min(5, x.shape[0])
            all_images = [ ('x', x), ('inpaint', inpaint), ('mask', mask), ('reference', reference), ('ref_imgs_4unet', ref_imgs_4unet),
              ('z4_gt',z[:,:3]),('z4_inpaint', z_inpaint[:,:3]),('tgt_mask_64', mask_resize),('z_ref',None if z_ref is None else z_ref[:,:3]),('ref_mask_64',ref_mask_64),]
            all_path = debug_dir / f"{str_t_pid()}.jpg"
            vis_tensors_A(all_images, all_path, vis_batch_size)

        if 1:
            assert self.model.conditioning_key is not None
            assert self.first_stage_key=='inpaint'
            assert self.cond_stage_key=='image'
        return {
            **batch,
            'z9': z_new,# b,9/14,...
            'z4_gt': z,
            'z4_inpaint': z_inpaint,
            #
            'tgt_mask_64': mask_resize,
            'ref_mask_64': ref_mask_64,
            #
            'z_ref': z_ref, # 'z_ref' is ambiguous but kept for legacy usage; hard-code the intended meaning
            #
            'landmarks': landmarks, # projected features, not raw coordinates
        }
        
    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                if self.first_stage_key=='inpaint':
                    return self.first_stage_model.decode(z[:,:4,:,:])
                else:
                    return self.first_stage_model.decode(z)

   

    # same as above but without decorator
    def differentiable_decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):  
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                df = self.split_input_params["vqf"]
                self.split_input_params['original_image_size'] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                output_list = [self.first_stage_model.encode(z[:, :, :, :, i])
                               for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:
                return self.first_stage_model.encode(x)
        else:
            return self.first_stage_model.encode(x)

    def get_input_and_conditioning(self,batch, device=None):
        if device is not None: batch = recursive_to(batch, device)
        #------------------------from shared_step-------------------------
        get_referenceZ=(REFNET.ENABLE and REFNET.task2layerNum[global_.task]>0) or CH14
        batch = self.get_input_(batch, self.first_stage_key,get_referenceZ=get_referenceZ)
        #------------------------from shared_step -> forward-------------------------
        assert ( self.model.conditioning_key is not None ) and self.cond_stage_trainable
        c=self.conditioning_with_feat(batch['ref_imgs'],landmarks=batch['landmarks'],enInputs=batch['enInputs'])
        return batch,c
    def shared_step(self, batch, **kwargs):
        task = self.set_task(batch)
        if (REFNET.ENABLE and REFNET.task2layerNum[task]>0):
            self.model.bank.clear()
        batch, c = self.get_input_and_conditioning(batch)
        z9 = batch['z9']
        z_ref = batch['z_ref']
        gt512 = batch['GT']
        gt256 = batch.get('GT256',None)
        # del batch
        loss = self(z9, c,z_ref=z_ref,gt512=gt512,gt256=gt256,task=task,batch=batch,)
        return loss

    def forward(self, x, c, *args, **kwargs):
        task = kwargs['task']
        # c is the reference tensor; target shares the same shape
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        self.u_cond_prop=random.uniform(0, 1)
        if self.model.conditioning_key is not None:
            # assert c is not None
            if self.cond_stage_trainable: # yes
                pass
                    
            if self.shorten_cond_schedule:  # TODO: drop this option
                raise Exception
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))

        if self.u_cond_prop<self.u_cond_percent and self.training :
            return self.p_losses(x, self.learnable_vector[task].repeat(x.shape[0],1,1), t, *args, **kwargs)
        else:  #x:[4,9,64,64] c:[4,1,768] x: img,inpaint_img,mask after first stage c:clip embedding
            return self.p_losses(x, c, t, *args, **kwargs)
    
        

    def apply_model(self, x_noisy, t, cond, return_ids=False,return_features=False,
        z_ref=None,
    ):

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn' # -->c_crossattn
            cond = {key: cond}

        if hasattr(self, "split_input_params"):
            assert 0,'This branch should not execute in practice'
            assert len(cond) == 1  # todo can only deal with one conditioning atm
            assert not return_ids  
            ks = self.split_input_params["ks"]  # eg. (128, 128)
            stride = self.split_input_params["stride"]  # eg. (64, 64)

            h, w = x_noisy.shape[-2:]

            fold, unfold, normalization, weighting = self.get_fold_unfold(x_noisy, ks, stride)

            z = unfold(x_noisy)  # (bn, nc * prod(**ks), L)
            # Reshape to img shape
            z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]

            if self.cond_stage_key in ["image", "LR_image", "segmentation",
                                       'bbox_img'] and self.model.conditioning_key:  # todo check for completeness
                c_key = next(iter(cond.keys()))  # get key
                c = next(iter(cond.values()))  # get value
                assert (len(c) == 1)  # todo extend to list with more than one elem
                c = c[0]  # get element

                c = unfold(c)
                c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                cond_list = [{c_key: [c[:, :, :, :, i]]} for i in range(c.shape[-1])]

            elif self.cond_stage_key == 'coordinates_bbox':
                assert 'original_image_size' in self.split_input_params, 'BoudingBoxRescaling is missing original_image_size'

                # assuming padding of unfold is always 0 and its dilation is always 1
                n_patches_per_row = int((w - ks[0]) / stride[0] + 1)
                full_img_h, full_img_w = self.split_input_params['original_image_size']
                # as we are operating on latents, we need the factor from the original image size to the
                # spatial latent size to properly rescale the crops for regenerating the bbox annotations
                num_downs = self.first_stage_model.encoder.num_resolutions - 1
                rescale_latent = 2 ** (num_downs)

                # get top left positions of patches as conforming for the bbbox tokenizer, therefore we
                # need to rescale the tl patch coordinates to be in between (0,1)
                tl_patch_coordinates = [(rescale_latent * stride[0] * (patch_nr % n_patches_per_row) / full_img_w,
                                         rescale_latent * stride[1] * (patch_nr // n_patches_per_row) / full_img_h)
                                        for patch_nr in range(z.shape[-1])]

                # patch_limits are tl_coord, width and height coordinates as (x_tl, y_tl, h, w)
                patch_limits = [(x_tl, y_tl,
                                 rescale_latent * ks[0] / full_img_w,
                                 rescale_latent * ks[1] / full_img_h) for x_tl, y_tl in tl_patch_coordinates]
                # patch_values = [(np.arange(x_tl,min(x_tl+ks, 1.)),np.arange(y_tl,min(y_tl+ks, 1.))) for x_tl, y_tl in tl_patch_coordinates]

                # tokenize crop coordinates for the bounding boxes of the respective patches
                patch_limits_tknzd = [torch.LongTensor(self.bbox_tokenizer._crop_encoder(bbox))[None].to(self.device)
                                      for bbox in patch_limits]  # list of length l with tensors of shape (1, 2)
                print(patch_limits_tknzd[0].shape)
                # cut tknzd crop position from conditioning
                assert isinstance(cond, dict), 'cond must be dict to be fed into model'
                cut_cond = cond['c_crossattn'][0][..., :-2].to(self.device)

                adapted_cond = torch.stack([torch.cat([cut_cond, p], dim=1) for p in patch_limits_tknzd])
                adapted_cond = rearrange(adapted_cond, 'l b n -> (l b) n')
                adapted_cond = self.get_learned_conditioning(adapted_cond)
                adapted_cond = rearrange(adapted_cond, '(l b) n d -> l b n d', l=z.shape[-1])

                cond_list = [{'c_crossattn': [e]} for e in adapted_cond]

            else:
                cond_list = [cond for i in range(z.shape[-1])]  # Todo make this more efficient

            # apply model by loop over crops
            output_list = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
            assert not isinstance(output_list[0],
                                  tuple)  # todo cant deal with multiple model outputs check this never happens

            o = torch.stack(output_list, axis=-1)
            o = o * weighting
            # Reverse reshape to img shape
            o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
            # stitch crops together
            x_recon = fold(o) / normalization

        else:
            x_recon = self.model(x_noisy, t, **cond, return_features=return_features, z_ref=z_ref,
                task=self.task, _trainer=self.trainer,
            )
        if return_features:
            return x_recon
        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon


    def p_losses(self, x_start, cond, t, noise=None, z_ref=None, gt512=None, gt256=None, task=None,
        batch :dict = None,
    ):
    # def p_losses_face(self, x_start, cond, t, reference=None,noise=None,GT_tar=None,landmarks=None):
        # initialize MoE auxiliary loss to 0 to allow unconditional accumulation later
        global_.moe_aux_loss = torch.tensor(0.0, device=self.device)
        if self.first_stage_key == 'inpaint':
            # x_start=x_start[:,:4,:,:]
            noise = default(noise, lambda: torch.randn_like(x_start[:,:4,:,:]))
            if 1:
                x_noisy = self.q_sample(x_start=x_start[:,:4,:,:], t=t, noise=noise)
                x_noisy = torch.cat((x_noisy,x_start[:,4:,:,:]),dim=1)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
            if 1:
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        if z_ref is not None:
            assert self.first_stage_key == 'inpaint', 'Expected first_stage_key to be "inpaint"'
            """
            z_ref: b,4,...
            z_ref = concat [z_ref_noisy, z_ref, tensor_1c]
            tensor_1c is temporarily set to all zeros
            """
            z_ref_noisy = self.q_sample(x_start=z_ref, t=t, noise=torch.randn_like(z_ref))
            tensor_1c = torch.zeros((z_ref.shape[0], 1, z_ref.shape[2], z_ref.shape[3]), device=z_ref.device)
            if REFNET.CH9:
                z_ref = torch.cat([z_ref_noisy, z_ref, tensor_1c], dim=1)
        if 1:
            model_output = self.apply_model(x_noisy, t, cond, z_ref=z_ref, )

        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        if DDIM_losses:
            ########################
            t_new = torch.randint(self.num_timesteps-1, self.num_timesteps, (x_start.shape[0],), device=self.device).long().to(self.device)
            # t_new=torch.tensor(t_new).to(self.device)
            # noise_rec = default(noise, lambda: torch.randn_like(x_start[:,:4,:,:]))
            x_noisy_rec = self.q_sample(x_start=x_start[:,:4,:,:], t=t_new, noise=noise)
            x_noisy_rec = torch.cat((x_noisy_rec,x_start[:,4:,:,:]),dim=1)
            
            
            ddim_steps=self.Reconstruct_DDIM_steps
            n_samples=x_noisy_rec.shape[0]
            shape=(4,64,64)
            scale=5
            ddim_eta=0.0
            start_code=x_noisy_rec
            test_model_kwargs=None
            # t=t
            
            samples_ddim, sample_intermediates = self.sampler.sample_train(S=ddim_steps, # 4 (from Reconstruct_DDIM_steps in trian.yaml)
                                                    conditioning=cond,
                                                    batch_size=n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=None,
                                                    eta=ddim_eta,
                                                    x_T=start_code,
                                                    t=t_new,
                                                    z_ref=z_ref,
                                                    test_model_kwargs=test_model_kwargs)


        
            
            # x_samples_ddim= self.differentiable_decode_first_stage(samples_ddim)
    
            other_pred_x_0=sample_intermediates['pred_x0']
            len_inter = len(other_pred_x_0)
            printC("len_inter", len_inter )
            for i in range(len(other_pred_x_0)):
                other_pred_x_0[i]=self.differentiable_decode_first_stage(other_pred_x_0[i])
            # x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            # x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
            
            
            ###########################################
            
            ID_loss=0
            clip_loss=0
            loss_lpips=0
            loss_rec=0
            loss_landmark=0
            
            # model_output=samples_ddim
            if 1:
                
                # x_samples_ddim=TF.resize(x_samples_ddim,(256,256))
                if 0:
                    inpaint_mask_64 = x_start[:,8,:,:] # inpaint region is 1, background is 0; shape b,64,64
                    masks=TF.resize(inpaint_mask_64,(other_pred_x_0[0].shape[2],other_pred_x_0[0].shape[3])) # b,512,512
                    if not 1:
                        masks = 1 - masks
                    #mask x_samples_ddim
                    x_samples_ddim_masked=[x_samples_ddim_preds*masks.unsqueeze(1) for x_samples_ddim_preds in other_pred_x_0]
                    # x_samples_ddim_masked=un_norm_clip(x_samples_ddim_masked)
                    # x_samples_ddim_masked = TF.normalize(x_samples_ddim_masked, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                else:
                    x_samples_ddim_masked = other_pred_x_0
                Landmark_loss_weight = 0
                ID_loss_weight = [0.3, 0, 0.1, 0.2, ][task]
                if ID_loss_weight > 0 :
                    ID_Losses=[]
                    for step,x_samples_ddim_preds in enumerate(x_samples_ddim_masked):
                        ID_loss,sim_imp,_=self.face_ID_model(x_samples_ddim_preds,gt512,clip_img=False)
                        ID_Losses.append(ID_loss)
                        loss_dict.update({f'{prefix}/ID_loss_{step}': ID_loss})
                    
                    ID_loss=torch.mean(torch.stack(ID_Losses))  
                    loss_dict.update({f'{prefix}/ID_loss': ID_loss})
                    loss_dict.update({f'{prefix}/sim_imp': sim_imp})
                
                CLIP_loss_weight = [1.5/4, 0.8, 1, 0.5, ][task]
                if CLIP_loss_weight > 0 :
                    def _loss(_img1,_img2):
                        _e1 = self.encoder_clip_face.forward_vit(_img1,resize=True)
                        _e2 = self.encoder_clip_face.forward_vit(_img2,resize=True)
                        return torch.nn.functional.mse_loss( _e1, _e2 )
                    clip_Losses=[]
                    for step,x_samples_ddim_preds in enumerate(x_samples_ddim_masked):
                        clip_loss = _loss(x_samples_ddim_preds,gt512)
                        clip_Losses.append(clip_loss)
                        loss_dict.update({f'{prefix}/clip_loss_{step}': clip_loss})
                    clip_loss=torch.mean(torch.stack(clip_Losses))  
                    loss_dict.update({f'{prefix}/clip_loss': clip_loss})
                
                LPIPS_loss_weight = [0.05, 0.015, 0.015, 0.015, ][task]
                if LPIPS_loss_weight>0:
                    if gt256 is not None:
                        _lpips_base_size = 256
                        _gt_for_lpips = gt256
                    else:
                        _lpips_base_size = 512
                        _gt_for_lpips = gt512
                    
                    for j in range(len(other_pred_x_0)):
                        for i in range(3):
                            _size = _lpips_base_size//2**i
                            _pred_for_lpips = F.adaptive_avg_pool2d(other_pred_x_0[j],(_size,_size))
                            _gt_for_lpips_resized = F.adaptive_avg_pool2d(_gt_for_lpips,(_size,_size))
                            loss_lpips_1 = self.lpips_loss(
                                _pred_for_lpips, 
                                _gt_for_lpips_resized,
                            )
                            loss_dict.update({f'{prefix}/loss_lpips_{j}_{i}': loss_lpips_1})
                            printC(f"loss_lpips_1 at {j} {i} :", loss_lpips_1)
                            loss_lpips += loss_lpips_1
                    loss_dict.update({f'{prefix}/loss_lpips': loss_lpips})
                
                REC_loss_weight = [0.05, 0.01, 0.01, 0.01, ][task]
                if REC_loss_weight > 0 : # rec loss
                    for j in range(len(other_pred_x_0)):
                        loss_rec_1 = torch.nn.functional.mse_loss( other_pred_x_0[j], gt512)
                        loss_dict.update({f'{prefix}/loss_rec_{j}': loss_rec_1})
                        printC(f"loss_rec_1 at {j} :", loss_rec_1)
                        loss_rec += loss_rec_1
                    loss_dict.update({f'{prefix}/loss_rec': loss_rec})
        if 1:
            if self.parameterization == "x0":
                target = x_start
            elif self.parameterization == "eps":
                target = noise
            else:
                raise NotImplementedError()
        
            # this should be an MSE loss
            loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
            loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
            loss_dict.update({f'{prefix}/loss_simple-t{task}': loss_simple.mean()})

            self.logvar = self.logvar.to(self.device)
            logvar_t = self.logvar[t].to(self.device)
            loss = loss_simple / torch.exp(logvar_t) + logvar_t
            # loss = loss_simple / torch.exp(self.logvar) + self.logvar
            if self.learn_logvar:
                loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
                loss_dict.update({'logvar': self.logvar.data.mean()})

            loss = self.l_simple_weight * loss.mean()

            loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3)) #??
            loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
            loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
            loss_dict.update({f'{prefix}/loss_vlb-t{task}': loss_vlb})
            loss += (self.original_elbo_weight * loss_vlb)
        else:
            loss = 0
        if DDIM_losses:
            _item = lambda _a:    _a.detach().cpu().item() if isinstance(_a,torch.Tensor) else _a 
            printC("orig, ID clip, lpips rec lmk:",
                f"{_item(loss):.4f}, {_item(ID_loss):.4f} {_item(clip_loss):.4f}, {_item(loss_lpips):.4f} {_item(loss_rec):.4f} {_item(loss_landmark):.4f}",
                f"{ID_Losses=}" if ID_loss_weight>0 else "",
                f"{clip_Losses=}" if CLIP_loss_weight>0 else "",
            )
            loss+=ID_loss_weight*ID_loss+LPIPS_loss_weight*loss_lpips+Landmark_loss_weight*loss_landmark+REC_loss_weight*loss_rec+CLIP_loss_weight*clip_loss

        # incorporate MoE auxiliary loss
        moe_aux = global_.moe_aux_loss
        if isinstance(moe_aux, torch.Tensor):
            loss = loss + moe_aux
            loss_dict.update({f'{prefix}/moe_aux_loss': moe_aux})
        loss_dict.update({f'{prefix}/loss': loss})
        loss_dict.update({f'{prefix}/loss-t{task}': loss})
        return loss, loss_dict



    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        
        if self.partial_training:# no
        # if True:
            print("Partial training.............................")
            train_names=self.trainable_keys
            train_names=[ 'attn2','norm2']
            params_train=[]
            for name,param in self.model.named_parameters():
                if "diffusion_model" not in name and param.requires_grad:
                    print(name)
                    params_train.append(param)
                    
                elif "diffusion_model" in name and any(train_name in name for train_name in train_names):
                    print(name)
                    params_train.append(param)
            params=params_train
        print("Setting up Adam optimizer.......................")

        if self.cond_stage_trainable:# yes
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            if hasattr(self,'encoder_clip_face'):
                params += list(self.encoder_clip_face.final_ln2.parameters())+list(self.encoder_clip_face.mapper2.parameters())
                if self.USE_proj_out_source:
                    params += list(self.proj_out_source__face.parameters())
            if hasattr(self,'encoder_clip_hair'):
                params += list(self.encoder_clip_hair.final_ln2.parameters())+list(self.encoder_clip_hair.mapper2.parameters())
                if self.USE_proj_out_source:
                    params += list(self.proj_out_source__hair.parameters())
            if hasattr(self,'encoder_clip_head_t2'):
                params += list(self.encoder_clip_head_t2.final_ln2.parameters())+list(self.encoder_clip_head_t2.mapper2.parameters())
            if hasattr(self,'encoder_clip_head_t3'):
                params += list(self.encoder_clip_head_t3.final_ln2.parameters())+list(self.encoder_clip_head_t3.mapper2.parameters())
            if hasattr(self,'encoder_clip_head_t2') or hasattr(self,'encoder_clip_head_t3'):
                if self.USE_proj_out_source:
                    params += list(self.proj_out_source__head.parameters())
            if hasattr(self,'ID_proj_out'):
                params += list(self.ID_proj_out.parameters())
            if hasattr(self,'landmark_proj_out'): # fixLmkProj
                params += list(self.landmark_proj_out.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        params.extend(self.learnable_vector)
        params = [p for p in params if p.requires_grad]

        # Build param groups: MoE gate/expert use larger LR.
        # Also apply per-task LR factor to all task-specific params.
        # only match MoE-related parameter names generated by the UNet wrappers
        moe_gate_ids = set()
        moe_ep_ids = set()
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if ".moe_gate_mlp." in name:
                moe_gate_ids.add(id(p))
            elif ".moe_experts_" in name:
                moe_ep_ids.add(id(p))

        params_ids = set(id(p) for p in params)
        task_specific_ids = set()
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) not in params_ids:
                continue
            is_task_specific = is_task_specific_(name)
            if rank_==0: print(f"{is_task_specific=} {name}")
            if is_task_specific:
                task_specific_ids.add(id(p))

        base_params = []
        task_specific_params = []
        moe_gate_params = []
        moe_ep_params = []
        for p in params:
            pid = id(p)
            if pid in task_specific_ids:
                task_specific_params.append(p)
            elif pid in moe_gate_ids:
                moe_gate_params.append(p)
            elif pid in moe_ep_ids:
                moe_ep_params.append(p)
            else:
                base_params.append(p)

        param_groups = []
        if base_params:
            param_groups.append({"params": base_params, "lr": lr})
        if task_specific_params:
            param_groups.append({"params": task_specific_params, "lr": lr * LR_factor})
        if moe_gate_params:
            param_groups.append({"params": moe_gate_params, "lr": lr * MOE_GATE_LR_MULT})
        if moe_ep_params:
            param_groups.append({"params": moe_ep_params, "lr": lr * MOE_EP_LR_MULT})
        if ZERO1_ENABLE:
            zero_pg = None
            if 1:
                if dist.is_available() and dist.is_initialized():
                    zero_pg = dist.new_group(backend='gloo')
            opt = ZeroRedundancyOptimizer(
                param_groups if (task_specific_params or moe_gate_params or moe_ep_params) else params,
                optimizer_class=torch.optim.AdamW if ADAM_or_SGD else torch.optim.SGD,
                lr=lr,
                process_group=zero_pg,
            )
        else:
            if ADAM_or_SGD:
                opt = torch.optim.AdamW(param_groups if (task_specific_params or moe_gate_params or moe_ep_params) else params, lr=lr)
            else:
                opt = torch.optim.SGD(param_groups if (task_specific_params or moe_gate_params or moe_ep_params) else params, lr=lr, momentum=0.9)
        if gate_('LatentDiffusion.configure_optimizers params:'):
            if (task_specific_params or moe_gate_params or moe_ep_params):
                print(f"base/task_specific/ep/gate lens: {len(base_params)=} {len(task_specific_params)=} {len(moe_ep_params)=} {len(moe_gate_params)=}")
                print(f"sum of .numel(): base={sum(p.numel() for p in base_params)} task_specific={sum(p.numel() for p in task_specific_params)} ep={sum(p.numel() for p in moe_ep_params)} gate={sum(p.numel() for p in moe_gate_params)}")
            else:
                print(f"{len(params)=}")
                print(f"sum of .numel(): {sum(param.numel() for param in params)}")
        if self.use_scheduler:# yes
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    def on_train_epoch_start(self):
        def _set_req_grad(p, flag):
            if p.requires_grad != flag:
                p.requires_grad = flag
                return 1
            return 0
        return
        if 0:
            train_now = self.current_epoch < N_EPOCHS_TRAIN_REF_AND_MID
        else: # alternating freezing
            train_now = (self.current_epoch % 2 == 0)
        ct_toggled = 0
        # 1) freeze all shared if not train_now; unfreeze when train_now
        ct_shared = 0
        for name, p in self.model.diffusion_model.named_parameters():
            # target only the shared weights inside Shared+LoRA wrappers: FFN.shared_ffn.* and Conv.shared.*
            is_shared = ('.shared_ffn.' in name) or ('.shared.' in name)
            if is_shared:
                ct_shared += _set_req_grad(p, train_now)
        print(f"[freeze@epoch]{self.current_epoch=} {train_now=} {ct_toggled=} {ct_shared=}")

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x
    def __repr__(self):
        if DEBUG:    return 'LatentDiffusion.__repr__'
        return super().__repr__()
    @property
    def model_size(self):
        if DEBUG:  return -1
        return super().model_size


from .bank import Bank
class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        diff_model_config['params']['is_refNet'] = False
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']
        if REFNET.ENABLE:
            diff_model_config_refNet = diff_model_config
            print('instantiate / deepcopy diffusion_model_refNet ing...')
            if 1:
                diff_model_config_refNet['params']['in_channels'] = 9 if REFNET.CH9 else 4
                diff_model_config_refNet['params']['is_refNet'] = True
                self.diffusion_model_refNet :UNetModel = instantiate_from_config(diff_model_config_refNet)
            else:
                self.diffusion_model_refNet :UNetModel = copy.deepcopy(self.diffusion_model) # faster than re-instantiating
                self.diffusion_model_refNet.is_refNet = True
            if 1:
                # print(f"before del: {len(self.diffusion_model_refNet.input_blocks)=}")
                if 1:
                    self.diffusion_model_refNet.input_blocks = self.diffusion_model_refNet.input_blocks[:9]
                del self.diffusion_model_refNet.middle_block
                del self.diffusion_model_refNet.output_blocks
                del self.diffusion_model_refNet.out
            print('over.')
            # Keep only a single diffusion_model_refNet; no t-suffixed clones

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None,return_features=False,
        z_ref=None,
        task = None,
        _trainer :pl.Trainer = None,
    ):
        _in_train_or_val = ( _trainer is not None ) and ( _trainer.validating or  _trainer.sanity_checking ) # indicates train or validation state
        assert self.conditioning_key == 'crossattn'
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)  #-->cc.shape = (bs, 1, 768) ## adding return_features  here only for testing
            if (REFNET.ENABLE and REFNET.task2layerNum[task]>0):
                if task in (0,2,3,):
                    cc_ref = cc[:,:-1, :]
                else:
                    cc_ref = cc
                printC("c for refNet",f"{custom_repr_v3(cc_ref)}")
                self.diffusion_model_refNet(z_ref, t, context=cc_ref,return_features=False)
            out = self.diffusion_model(x, t, context=cc,return_features=return_features)
            if (REFNET.ENABLE and REFNET.task2layerNum[task]>0) and not (self.training or _in_train_or_val):
            # if 1:
                self.bank.clear()
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out  #-->out.shape = (bs, 4,64,64)
