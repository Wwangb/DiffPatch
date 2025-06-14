import torch
import random
import numpy as np
import os
import time
import json
import glob
import torch.nn.functional as F

from tensorboardX import SummaryWriter

from contextlib import nullcontext
from torch import optim, autograd
from torch.cuda.amp import autocast

from PIL import Image
from tqdm import tqdm
from diffusers import DDIMScheduler, StableDiffusionPipeline
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
from models.common import DetectMultiBackend
from test_patch import PatchTester
from utils.common import IMG_EXTNS, pad_to_square
from utils.config_parser import load_config_object, get_argparser
from utils.patch import PatchApplier, PatchTransformer
from utils.loss import MaxProbExtractor, TotalVariationLoss, Detection_CrossEntropy, Detection_Loss
from utils.general import non_max_suppression, xyxy2xywh
from utils.dataset import YOLODataset
from torchvision import transforms as T
from typing import Union


class PatchTrainer:
    """
    Module for training on dataset to generate adv patches
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.dev = cfg.device
        self.SEED = cfg.seed
        if self.SEED is not None:
            random.seed(self.SEED)
            np.random.seed(self.SEED)
            torch.manual_seed(self.SEED)
            torch.cuda.manual_seed(self.SEED)
        
        # setting benchmark to False reduces training time for our setup
        torch.backends.cudnn.benchmark = False

        # attack model
        detect_model = DetectMultiBackend(cfg.weights_file, device=self.dev, dnn=False, data=None, fp16=False)
        self.detect_model = detect_model.eval()

        # generate model
        pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.model_path,
        )
        self.vae = pipe.vae.to(self.dev)
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.to(self.dev)
        self.unet = pipe.unet.to(self.dev)
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.scheduler.set_timesteps(50)
        # self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(self.dev)
        # self.clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
        # self.clip_model.eval()

        # freeze model parameter
        for param in self.detect_model.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.unet.parameters():
            param.requires_grad = False
        
        self.patch_transformer = PatchTransformer(
            cfg.target_size_frac, cfg.mul_gau_mean, cfg.mul_gau_std, cfg.x_off_loc, cfg.y_off_loc, self.dev).to(self.dev)
        self.patch_applier = PatchApplier(cfg.patch_alpha).to(self.dev)
        # self.tv_loss = TotalVariationLoss().to(self.dev)
        # self.detect_loss = MaxProbExtractor(cfg).to(self.dev)
        # self.detect_loss = Detection_CrossEntropy(cfg).to(self.dev)
        self.detect_loss = Detection_Loss(cfg).to(self.dev)

        # set log dir
        cfg.log_dir = os.path.join(cfg.log_dir, f'{time.strftime("%Y%m%d-%H%M%S")[4:8]}_{cfg.patch_name}')
        self.writer = SummaryWriter(cfg.log_dir)
        # save config parameters to tensorboard logs
        for cfg_key, cfg_val in cfg.items():
            self.writer.add_text(cfg_key, str(cfg_val))
        
        # setting train image augmentations
        transforms = None
        if cfg.augment_image:
            transforms = T.Compose(
                [T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1)),
                T.ColorJitter(brightness=.2, hue=.04, contrast=.1),
                T.RandomAdjustSharpness(sharpness_factor=2)])
            
        self.train_loader = torch.utils.data.DataLoader(
            YOLODataset(image_dir=cfg.image_dir,
                        label_dir=cfg.label_dir,
                        max_labels=cfg.max_labels,
                        model_in_sz=cfg.model_in_sz,
                        use_even_odd_images=cfg.use_even_odd_images,
                        transform=transforms,
                        filter_class_ids=cfg.objective_class_id,
                        min_pixel_area=cfg.min_pixel_area,
                        shuffle=True
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.dev == "cuda" else False)
        self.epoch_length = len(self.train_loader)



    def load_512(self, image_path, left=0, right=0, top=0, bottom=0):
        if type(image_path) is str:
            image = np.array(Image.open(image_path))[:, :, :3]
        else:
            image = image_path
        h, w, c = image.shape
        left = min(left, w-1)
        right = min(right, w - left - 1)
        top = min(top, h - left - 1)
        bottom = min(bottom, h - top - 1)
        image = image[top:h-bottom, left:w-right]
        h, w, c = image.shape
        if h < w:
            offset = (w - h) // 2
            image = image[:, offset:offset + h]
        elif w < h:
            offset = (h - w) // 2
            image = image[offset:offset + w]
        image = np.array(Image.fromarray(image).resize((512, 512)))
        return image
    

    def load_and_preprocess_image(self, image_path):
        image = Image.open(image_path)
        image = self.clip_processor(image, return_tensors="pt")
        return image
    

    # image2latent
    @torch.no_grad()
    def image2latent(self, image):
        """
        a image to a latent 
        input size:
            image: 3x512x512
        output size:
            latent: 3x64x64
        """
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to('cuda')
                latents = self.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents
    

    @torch.no_grad()
    def _read_mask(self, mask_path):
        """
        load a patch mask
        output:
            mask: 64x64
            org_mask: 512x512
        """
        org_mask = Image.open(mask_path).convert("L")
        mask = org_mask.resize((64, 64), Image.NEAREST)
        mask = np.array(mask) / 255
        org_mask = np.array(org_mask) / 255
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        org_mask[org_mask < 0.5] = 0
        org_mask[org_mask >= 0.5] = 1
        mask = torch.from_numpy(mask).float().to(self.dev)
        org_mask = torch.from_numpy(org_mask).float().to(self.dev)

        return mask, org_mask
    

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.tokenizer(
            [""], padding="max_length", max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.dev))[0]
        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.dev))[0]
        context = torch.cat([uncond_embeddings, text_embeddings])
        return context
    

    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample


    @torch.no_grad()
    def ddim_inversion(self, image, context, sp=25):
        latent = self.image2latent(image)
        _, cond_embeddings = context.chunk(2)
        ddim_latents = [latent]
        latent = latent.clone().detach()
        for i in range(sp):
            t = self.scheduler.timesteps[len(self.scheduler.timesteps) - i - 1]
            noise_pred = self.unet(latent, t, encoder_hidden_states=cond_embeddings)["sample"]
            latent = self.next_step(noise_pred, t, latent)
            ddim_latents.append(latent)
        return ddim_latents


    def get_noise_pred(self, latents, t, context):
        latents_input = torch.cat([latents] * 2)
        noise_pred = self.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + 7.5 * (noise_prediction_text - noise_pred_uncond)
        latents = self.prev_step(noise_pred, t, latents)
        return latents
    

    def null_optimization(self, latents, context, num_inner_steps, epsilon, sp=25):
        uncond_embeddings, cond_embeddings = context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * sp)
        for i in range(sp):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = optim.Adam([uncond_embeddings], lr=1e-2 * (1. - (i + 50 - sp) / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.scheduler.timesteps[i + 50 - sp]
            with torch.no_grad():
                noise_pred_cond = self.unet(latent_cur, t, encoder_hidden_states=cond_embeddings)["sample"]
            for j in range(num_inner_steps):
                noise_pred_uncond = self.unet(latent_cur, t, encoder_hidden_states=uncond_embeddings)["sample"]
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = F.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, context)
        bar.close()
        return uncond_embeddings_list


    def make_source_latent(self, image, prompt, num_inner_steps=10, early_stop_epsilon=1e-5):
        context = self.init_prompt(prompt)
        print("DDIM inversion...")
        ddim_latents = self.ddim_inversion(image, context)
        print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, context, num_inner_steps, early_stop_epsilon)
        return ddim_latents[-1], uncond_embeddings
    

    @torch.no_grad()
    def get_text_embedding(self, prompt):

        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.dev))[0]
        return text_embeddings


    def gen_adv_patch(self, text_embeddings, uncond_embeddings, latent, guidance_scale=7.5, start_time=25):
        batch_size = 1
        height = width = 512

        latents = latent.expand(batch_size, self.unet.config.in_channels, height // 8, width // 8).to(self.dev)
        
        # for i, t in enumerate(tqdm(self.scheduler.timesteps[-start_time:])):
        for i, t in enumerate(self.scheduler.timesteps[-start_time:]):
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])

            latents_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(
                        latents_model_input, t, encoder_hidden_states=context
                )["sample"]
            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            latents = self.prev_step(noise_pred, t, latents)

        #     # compute the previous noisy sample x_t -> x_t-1
            # latent = self.scheduler.step(noise_pred, t, latent).prev_sample
        
        latents = 1 / 0.18215 * latents
        adv_patch = self.vae.decode(latents)["sample"]
        adv_patch = (adv_patch / 2 + 0.5).clamp(0, 1)
        adv_patch = adv_patch.squeeze(0)

        return adv_patch
    

    def train(self):

        # output dirs
        patch_dir = os.path.join(self.cfg.log_dir, "patches")
        log_file_name = os.path.join(self.cfg.log_dir, "metrics.json")
        os.makedirs(patch_dir, exist_ok=True)
        for img_dir in ["train_patch_applied_imgs", "val_clean_imgs", "val_patch_applied_imgs"]:
            os.makedirs(os.path.join(self.cfg.log_dir, img_dir), exist_ok=True)
        
        # dump cfg json file
        with open(os.path.join(self.cfg.log_dir, "cfg.json"), 'w', encoding='utf-8') as json_f:
            json.dump(self.cfg, json_f, ensure_ascii=False, indent=4)
        
        # fix loss targets
        loss_target = self.cfg.loss_target
        if loss_target == "obj":
            self.cfg.loss_target = lambda obj, cls: obj
        elif loss_target == "cls":
            self.cfg.loss_target = lambda obj, cls: cls
        elif loss_target in {"obj * cls", "obj*cls"}:
            self.cfg.loss_target = lambda obj, cls: obj * cls
        else:
            raise NotImplementedError(
                f"Loss target {loss_target} not been implemented")
    
        source_image = self.load_512(self.cfg.image_path, *self.cfg.image_offsets)
        text_embeddings = self.get_text_embedding([self.cfg.prompt])
        adv_latent, uncond_embeddings = self.make_source_latent(source_image, self.cfg.prompt)

        mask, org_mask = self._read_mask(self.cfg.mask_path)
        source_latent = adv_latent.clone()
        if self.cfg.is_restraint:
            up_scale = source_latent + self.cfg.res_eps
            down_scale = source_latent - self.cfg.res_eps

        adv_latent.requires_grad = True
        optimizer = optim.Adam([adv_latent], lr=self.cfg.start_lr, amsgrad=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)
        start_val_state = True

        start_time = time.time()

        for epoch in range(1, self.cfg.n_epochs + 1):

            out_patch_path = os.path.join(patch_dir, f"e_{epoch}.png")
            out_latent_path = os.path.join(patch_dir, f"latent_{epoch}.pt")
            
            # set loss
            # min_tv_loss = torch.tensor(self.cfg.min_tv_loss, device=self.dev)
            zero_tensor = torch.tensor([0], device=self.dev)
            ep_loss = 0

            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(self.train_loader),
                                                        desc=f'Running train epoch {epoch}',
                                                        total=self.epoch_length):
                with autograd.set_detect_anomaly(mode=True):

                    # if self.cfg.is_restraint:
                    #     adv_latent_clamp = adv_latent.clamp(down_scale, up_scale)
                    # else:
                    #     adv_latent_clamp = adv_latent
                    
                    adv_patch = self.gen_adv_patch(text_embeddings, uncond_embeddings, adv_latent)
                    # save patch
                    adv_patch_cpu = adv_patch.detach().cpu()
                    adv_patch_save = adv_patch.detach().cpu()
                    image_save = adv_patch_cpu.permute(1,2,0).numpy()
                    image_save = (image_save * 255).round().astype("uint8")
                    Image.fromarray(image_save).save(self.cfg.now_patch_path)

                    # source_preprocess_image = self.load_and_preprocess_image(self.cfg.image_path)["pixel_values"].to(self.dev)
                    # source_image_embedding = self.clip_model.get_image_features(source_preprocess_image)
                    # now_preprocess_image = self.load_and_preprocess_image(self.cfg.now_patch_path)["pixel_values"].to(self.dev)
                    # now_image_embedding = self.clip_model.get_image_features(now_preprocess_image)
                    # sim_loss = 1 - F.cosine_similarity(source_image_embedding, now_image_embedding)
                    # sim_loss = 1 - F.cosine_similarity(source_preprocess_image, now_preprocess_image)

                    adv_patch = adv_patch * org_mask
                    # save cropped patch
                    adv_patch_cpu = adv_patch.detach().cpu()
                    image_save = adv_patch_cpu.permute(1,2,0).numpy()
                    image_save = (image_save * 255).round().astype("uint8")
                    Image.fromarray(image_save).save(self.cfg.now_cropped_path)

                    if(start_val_state):
                        with torch.no_grad():
                            start_val_state = False
                            self.val(0, self.cfg.now_cropped_path, log_dir=log_file_name)

                    img_batch = img_batch.to(self.dev, non_blocking=True)
                    lab_batch = lab_batch.to(self.dev, non_blocking=True)
                    adv_batch_t = self.patch_transformer(
                        adv_patch, lab_batch, org_mask, self.cfg.model_in_sz,
                        use_mul_add_gau=self.cfg.use_mul_add_gau,
                        do_transforms=self.cfg.transform_patches,
                        do_rotate=self.cfg.rotate_patches,
                        rand_loc=self.cfg.random_patch_loc)
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                    p_img_batch = F.interpolate(p_img_batch, (self.cfg.model_in_sz[0], self.cfg.model_in_sz[1]))

                    img = p_img_batch[0, :, :, ]
                    img = T.ToPILImage()(img.detach().cpu())
                    img.save(os.path.join(self.cfg.log_dir, "train_patch_applied_imgs", f"b_{i_batch}.jpg"))

                    with autocast() if self.cfg.use_amp else nullcontext():
                        output = self.detect_model(p_img_batch)[0]
                        # tv = self.tv_loss(adv_patch) if self.cfg.tv_mult != 0 else zero_tensor
                        # max_prob = self.prob_extractor(output)
                        detection_loss = self.detect_loss(output, lab_batch)
                        # detection_loss = self.detect_loss(output)

                    # det_loss = torch.mean(detection_loss)
                    det_loss = torch.mean(detection_loss)
                    # tv_loss = torch.max(tv * self.cfg.tv_mult, min_tv_loss)

                    loss = det_loss
                    # loss = det_loss + tv_loss
                    ep_loss += loss

                    loss.backward()
                    with torch.no_grad():
                        adv_latent.grad *= mask

                    optimizer.step()
                    if self.cfg.is_restraint:
                        adv_latent.data = adv_latent.data.clamp(down_scale, up_scale)
                    optimizer.zero_grad(set_to_none=True)

                    if i_batch % self.cfg.tensorboard_batch_log_interval == 0:
                        iteration = self.epoch_length * epoch + i_batch
                        self.writer.add_scalar(
                            "total_loss", loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar(
                            "loss/det_loss", det_loss.detach().cpu().numpy(), iteration)
                        # self.writer.add_scalar(
                        #     "loss/tv_loss", tv_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar(
                            "misc/epoch", epoch, iteration)
                        self.writer.add_scalar(
                            "misc/learning_rate", optimizer.param_groups[0]["lr"], iteration)
                        self.writer.add_image(
                            "patch", adv_patch_cpu, iteration)
                    if i_batch + 1 < len(self.train_loader):
                        # del adv_batch_t, output, max_prob, det_loss, p_img_batch, loss
                        del adv_batch_t, output, detection_loss, det_loss, p_img_batch, loss
            ep_loss = ep_loss / len(self.train_loader)
            scheduler.step(ep_loss)

            max_diff = torch.max(torch.abs(adv_latent - source_latent))
            with open('max_abs_diff.txt', 'a') as f:
                f.write("epoch: " + str(epoch) + "  " + str(max_diff.item()) + '\n')

            # save patch after every patch_save_epoch_freq epochs
            if epoch % self.cfg.patch_save_epoch_freq == 0:
                img = T.ToPILImage(self.cfg.patch_img_mode)(adv_patch_save)
                img.save(out_patch_path)
                torch.save(adv_latent, out_latent_path)
                # del adv_batch_t, output, max_prob, det_loss, p_img_batch, loss
                del adv_batch_t, output, detection_loss, det_loss, p_img_batch, loss

            # run validation to calc asr on val set if self.val_dir is not None
            if all([self.cfg.val_image_dir, self.cfg.val_epoch_freq]) and epoch % self.cfg.val_epoch_freq == 0:
                with torch.no_grad():
                    self.val(epoch, out_patch_path, log_dir=log_file_name)
        print(f"Total training time {time.time() - start_time:.2f}s")


    def val(self, epoch: int, patchfile: str, log_dir: str, conf_thresh: float = 0.5, nms_thresh: float = 0.5) -> None:
        """
        Calculates the attack success rate according for the patch with respect to different bounding box areas
        """
        # load patch from file
        patch_img = Image.open(patchfile).convert(self.cfg.patch_img_mode)
        patch_img = T.Resize(self.cfg.patch_size)(patch_img)
        adv_patch_cpu = T.ToTensor()(patch_img)
        adv_patch = adv_patch_cpu.to(self.dev)
        _, org_mask = self._read_mask(self.cfg.mask_path)

        img_paths = glob.glob(os.path.join(self.cfg.val_image_dir, "*"))
        img_paths = sorted([p for p in img_paths if os.path.splitext(p)[-1] in IMG_EXTNS])

        train_t_size_frac = self.patch_transformer.t_size_frac
        self.patch_transformer.t_size_frac = self.cfg.target_size_frac  # use a frac of 0.3 for validation
        # to calc confusion matrixes and attack success rates later
        all_labels = []
        all_patch_preds = []

        m_h, m_w = self.cfg.model_in_sz
        cls_id = self.cfg.objective_class_id
        zeros_tensor = torch.zeros([1, 5]).to(self.dev)
        #### iterate through all images ####
        for imgfile in tqdm(img_paths, desc=f'Running val epoch {epoch}'):
            img_name = os.path.splitext(imgfile)[0].split('/')[-1]
            img = Image.open(imgfile).convert('RGB')
            padded_img = pad_to_square(img)
            padded_img = T.Resize(self.cfg.model_in_sz)(padded_img)

            #######################################
            # generate labels to use later for patched image
            padded_img_tensor = T.ToTensor()(padded_img).unsqueeze(0).to(self.dev)
            pred = self.detect_model(padded_img_tensor)
            boxes = non_max_suppression(pred, conf_thresh, nms_thresh)[0]
            # if doing targeted class performance check, ignore non target classes
            if cls_id is not None:
                boxes = boxes[boxes[:, -1] == cls_id]
            all_labels.append(boxes.clone())
            boxes = xyxy2xywh(boxes)

            labels = []
            for box in boxes:
                cls_id_box = box[-1].item()
                x_center, y_center, width, height = box[:4]
                x_center, y_center, width, height = x_center.item(), y_center.item(), width.item(), height.item()
                labels.append([cls_id_box, x_center / m_w, y_center / m_h, width / m_w, height / m_h])

            # save img
            padded_img_drawn = PatchTester.draw_bbox_on_pil_image(
                all_labels[-1], padded_img, self.cfg.class_list)
            padded_img_drawn.save(os.path.join(self.cfg.log_dir, "val_clean_imgs", img_name + ".jpg"))

            # use a filler zeros array for no dets
            label = np.asarray(labels) if labels else np.zeros([1, 5])
            label = torch.from_numpy(label).float()
            if label.dim() == 1:
                label = label.unsqueeze(0)

            #######################################
            # Apply proper patches
            img_fake_batch = padded_img_tensor
            lab_fake_batch = label.unsqueeze(0).to(self.dev)

            if len(lab_fake_batch[0]) == 1 and torch.equal(lab_fake_batch[0], zeros_tensor):
                # no det, use images without patches
                p_tensor_batch = padded_img_tensor
            else:
                # transform patch and add it to image
                adv_batch_t = self.patch_transformer(
                    adv_patch, lab_fake_batch, org_mask, self.cfg.model_in_sz,
                    use_mul_add_gau=self.cfg.use_mul_add_gau,
                    do_transforms=self.cfg.transform_patches,
                    do_rotate=self.cfg.rotate_patches,
                    rand_loc=self.cfg.random_patch_loc)
                p_tensor_batch = self.patch_applier(img_fake_batch, adv_batch_t)

            pred = self.detect_model(p_tensor_batch)
            boxes = non_max_suppression(pred, conf_thresh, nms_thresh)[0]
            # if doing targeted class performance check, ignore non target classes
            if cls_id is not None:
                boxes = boxes[boxes[:, -1] == cls_id]
            all_patch_preds.append(boxes.clone())

            # save properly patched img
            p_img_pil = T.ToPILImage('RGB')(p_tensor_batch.squeeze(0).cpu())
            p_img_pil_drawn = PatchTester.draw_bbox_on_pil_image(
                all_patch_preds[-1], p_img_pil, self.cfg.class_list)
            p_img_pil_drawn.save(os.path.join(self.cfg.log_dir, "val_patch_applied_imgs", img_name + ".jpg"))

        # reorder labels to (Array[M, 5]), class, x1, y1, x2, y2
        all_labels = torch.cat(all_labels)[:, [5, 0, 1, 2, 3]]
        # patch and noise labels are of shapes (Array[N, 6]), x1, y1, x2, y2, conf, class
        all_patch_preds = torch.cat(all_patch_preds)
        asr_s, asr_m, asr_l, asr_a = PatchTester.calc_asr(
            all_labels, all_patch_preds,
            class_list=self.cfg.class_list,
            cls_id=cls_id)

        print(f"Validation metrics for images with patches:")
        print(f"\tASR@thres={conf_thresh}: asr_s={asr_s:.3f},  asr_m={asr_m:.3f},  asr_l={asr_l:.3f},  asr_a={asr_a:.3f}")

        self.writer.add_scalar("val_asr_per_epoch/area_small", asr_s, epoch)
        self.writer.add_scalar("val_asr_per_epoch/area_medium", asr_m, epoch)
        self.writer.add_scalar("val_asr_per_epoch/area_large", asr_l, epoch)
        self.writer.add_scalar("val_asr_per_epoch/area_all", asr_a, epoch)
        metrics = {
            'epoch': epoch,
            'area_all': asr_a,
            'area_small': asr_s,
            'area_medium': asr_m,
            'area_large': asr_l
        }
        with open(log_dir, 'a+') as f:
            json.dump(metrics, f)
            f.write('\n')
        del adv_batch_t, padded_img_tensor, p_tensor_batch
        torch.cuda.empty_cache()
        self.patch_transformer.t_size_frac = train_t_size_frac



def main():
    parser = get_argparser()
    args = parser.parse_args()
    cfg = load_config_object(args.config)
    trainer = PatchTrainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()