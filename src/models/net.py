import torch
from torch import nn
from torch.autograd import Variable

from .clip import clip
from .loralib import *
from .vit import FusionViT
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


_tokenizer = _Tokenizer()


def load_clip_to_cpu(model_name='ViT-L/14'):
    url = clip._MODELS[model_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model.float()



class LanguageGuidedAlignment(nn.Module):
    def __init__(self, clip_model, classnames=["real", "synthetic"], **kwargs):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 3
        ctx_init = 'a photo is'
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
        
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        
        # patch-basaed enhancer in LGA
        d_model = clip_model.ln_final.weight.shape[0]
        num_heads = d_model // 64
        d_ffn = d_model * 4
        self.patch_basaed_enhancer = nn.MultiheadAttention(d_model, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        # FFN in patch-basaed enhancer
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        

    def forward_FFN(self, tgt):
        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)
        return tgt
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
            
        
        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx                   # (n_ctx, ctx_dim)
        # patch-basaed enhancer
        tgt = ctx[:, None].repeat_interleave(im_features.shape[0], dim=1)
        tgt2 = self.patch_basaed_enhancer(tgt, im_features.transpose(0, 1), im_features.transpose(0, 1))[0]
        tgt = tgt + tgt2

        tgt = self.norm1(tgt)
        tgt = self.forward_FFN(tgt)

        ctx_shifted = tgt.transpose(0, 1)
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts
    
    
    
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, return_full=False):
        
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if return_full:
            x = x @ self.text_projection
        else:
            x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x
    

class TextSemanticEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text, return_full=False):
        x = self.token_embedding(text)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if return_full:
            x = x @ self.text_projection
        else:
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x
    
    
class ForgeryFeatureIdentity(nn.Module):
    
    def __init__(self, in_c, out_c, h_dim, num_classes=2, **kwargs):
        super().__init__()
        self.linear1 = nn.Linear(in_c, in_c//2)
        self.linear2 = nn.Linear(in_c//2, in_c)
        self.act = nn.ReLU(in_c)
        self.proj = nn.Linear(in_c, h_dim)
        self.fc = nn.Linear(h_dim, num_classes)
    
    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = x + residual
        x = self.proj(x)
        feat = x
        x = self.fc(x)
        return x, feat
    
    
    
class LGMLoss(nn.Module):
    """
    LGMLoss whose covariance is fixed as Identity matrix
    """
    def __init__(self, num_classes, feat_dim, alpha=1.0):
        super(LGMLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.alpha = alpha

        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))


    def forward(self, feat, label=None):
        batch_size = feat.shape[0]

        diff = torch.unsqueeze(feat, dim=1) - torch.unsqueeze(self.centers, dim=0)
        diff = torch.mul(diff, diff)
        dist = torch.sum(diff, dim=-1)
        
        if self.training:
            label = label.to(torch.long)
            y_onehot = torch.FloatTensor(batch_size, self.num_classes)
            y_onehot.zero_()
            y_onehot = Variable(y_onehot).cuda()
            y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.alpha)
            y_onehot = y_onehot + 1.0
            margin_dist = torch.mul(dist, y_onehot)
            margin_logits = -0.5 * margin_dist
            logits = -0.5 * dist
            cdiff = feat - torch.index_select(self.centers, dim=0, index=label.long())
            likelihood = (1.0/batch_size) * cdiff.pow(2).sum(1).sum(0) / 2.0
            return logits, margin_logits, likelihood
        else:
            logits = -0.5 * dist
            return logits, None, None


class BaseModel(nn.Module):
    
    def __init__(self, model_name, class_name, **kwargs):
        super().__init__()
        self.cfg = {
            'backbone': model_name,
            'position': 'all',
            'encoder': 'vision',
            'params': ['q', 'k', 'v'],
            'r': 2,
            'alpha': 1,
            'dropout_rate': 0.25,
            'classnames': class_name,
            'num_layer_vit': 8,
            'num_classes': 1,
            'dim': 256,
            

        }
        self.clip_model = load_clip_to_cpu(self.cfg['backbone'])
        self.clip_model.eval()
        
        self.list_lora_layers = apply_lora(self.cfg, self.clip_model)
    
        self.image_encoder = self.clip_model.visual
        self.dtype = self.clip_model.dtype
        self.logit_scale = self.clip_model.logit_scale
        self.d_model = self.clip_model.ln_final.weight.shape[0]
        self.layers = self.image_encoder.layers
        self.width =self.image_encoder.width
        
        self.text_encoder = TextEncoder(self.clip_model)
        self.semantic_text_encoder = TextSemanticEncoder(self.clip_model)
        
        # self.dim = self.width // 2
        self.dim = self.cfg['dim']
        
        # self.classifcation_head = FusionViT(num_patches=self.layers, num_classes=self.cfg['num_classes'], in_dim=self.width, dim=self.dim, depth=6, heads=self.dim//6, mlp_dim=self.dim * 2, dropout = 0.1)
        # self.classifcation_head = FusionViT(num_patches=self.layers, num_classes=self.cfg['num_classes'], in_dim=self.width, dim=self.dim, depth=6, heads=self.dim//4, mlp_dim=self.dim * 2, dropout = 0.1)
        # self.classifcation_head = FusionViT(num_patches=self.layers, num_classes=self.cfg['num_classes'], in_dim=self.width, dim=self.dim, depth=6, heads=self.dim//6, mlp_dim=self.dim * 2, dropout = 0.1) # 384
        # self.classifcation_head = FusionViT(num_patches=self.layers, num_classes=self.cfg['num_classes'], in_dim=self.width, dim=self.dim, depth=6, heads=self.dim//8, mlp_dim=self.dim * 2, dropout = 0.1) # 512
        self.classifcation_head = FusionViT(num_patches=self.layers, num_classes=self.cfg['num_classes'], in_dim=self.width, dim=self.dim, depth=8, heads=self.dim//4, mlp_dim=self.dim * 2, dropout = 0.1) # 128
        
        
        
    def load_lora_weight(self, lora_path):
        self.list_lora_layers = load_lora(self.cfg, self.list_lora_layers, lora_path)
    
        
    def save_lora_weight(self, lora_path):
        save_lora(self.cfg, self.list_lora_layers, lora_path)
    
        
    def forward(self, images, text=None, label=None):
        images = images.to(self.dtype)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            image_features, hidden_features = self.image_encoder(images, return_full=False, return_hidden=True)
            logits = self.classifcation_head(hidden_features)
            
        if not self.training:
            return logits
        
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            semantic_text_feats = self.semantic_text_encoder(text)
            cosine_similarity = logit_scale * image_features_norm @ semantic_text_feats.t()
        return logits, cosine_similarity
    
    
    
if __name__ == '__main__':
    
    inputs = torch.randn((32, 3, 224, 224)).to('cuda')
    net = BaseModel()
    
    net.to('cuda')
    for i in range(100):
        outputs = net(inputs)
        print(i, outputs.shape)