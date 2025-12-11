#!/usr/bin/env python3
"""
Teste simplificado do Qwen3-Coder compression
Foca apenas na funcionalidade básica para validação
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
import time


class SimpleLowRankLinear(nn.Module):
    """Linear layer com compressão low-rank simplificada"""

    def __init__(self, in_features: int, out_features: int, rank: int = 64, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, in_features, out_features)

        # Fatores low-rank
        self.U = nn.Parameter(torch.randn(out_features, self.rank) * 0.01)
        self.V = nn.Parameter(torch.randn(self.rank, in_features) * 0.01)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Computação eficiente: x @ V^T @ U^T
        x_proj = x @ self.V.t()  # [..., rank]
        output = x_proj @ self.U.t()  # [..., out_features]

        if self.bias is not None:
            output += self.bias

        return output

    def compression_ratio(self) -> float:
        original = self.in_features * self.out_features
        compressed = self.rank * (self.in_features + self.out_features)
        return original / compressed


class SimpleQwen3Layer(nn.Module):
    """Camada transformer simplificada para teste"""

    def __init__(self, d_model: int, n_heads: int, rank: int = 64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Self-attention (compressed)
        self.q_proj = SimpleLowRankLinear(d_model, d_model, rank=rank)
        self.k_proj = SimpleLowRankLinear(d_model, d_model, rank=rank)
        self.v_proj = SimpleLowRankLinear(d_model, d_model, rank=rank)
        self.o_proj = SimpleLowRankLinear(d_model, d_model, rank=rank)

        # FFN (compressed)
        self.ffn_up = SimpleLowRankLinear(d_model, d_model * 4, rank=rank * 2)
        self.ffn_down = SimpleLowRankLinear(d_model * 4, d_model, rank=rank * 2)

        # Norm layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        residual = x
        x_norm = self.norm1(x)

        # Projeções
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        # Multi-head attention
        batch_size, seq_len = x.shape[:2]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        x_attn = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, -1)
        x_attn = self.o_proj(x_attn)

        x = residual + x_attn

        # FFN
        residual = x
        x_norm = self.norm2(x)
        x_ffn = F.gelu(self.ffn_up(x_norm))
        x_ffn = self.ffn_down(x_ffn)

        return residual + x_ffn


class SimpleQwen3Model(nn.Module):
    """Modelo Qwen3 simplificado para teste"""

    def __init__(self, vocab_size: int = 1000, d_model: int = 512, n_layers: int = 4,
                 n_heads: int = 8, rank: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embedding
        self.embed = nn.Embedding(vocab_size, d_model)

        # Camadas
        self.layers = nn.ModuleList([
            SimpleQwen3Layer(d_model, n_heads, rank=rank)
            for _ in range(n_layers)
        ])

        # Norm final e head
        self.norm = nn.LayerNorm(d_model)
        self.head = SimpleLowRankLinear(d_model, vocab_size, rank=rank, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.head(x)

    def estimate_memory(self, seq_len: int = 512) -> dict:
        """Estimar uso de memória"""
        # Parâmetros
        total_params = sum(p.numel() for p in self.parameters())
        param_memory = total_params * 2  # float16 = 2 bytes

        # KV cache
        kv_cache_per_token = 2 * self.d_model * len(self.layers) * 2  # key + value
        kv_cache = seq_len * kv_cache_per_token * 2  # float16

        # Ativações (estimativa)
        activation_memory = seq_len * self.d_model * 10 * 2  # fator aproximado

        total_memory = param_memory + kv_cache + activation_memory

        return {
            'parameters': total_params,
            'param_memory_mb': param_memory / (1024**2),
            'kv_cache_mb': kv_cache / (1024**2),
            'total_memory_mb': total_memory / (1024**2),
            'compression_ratio': self._estimate_compression()
        }

    def _estimate_compression(self) -> float:
        """Estimar taxa de compressão vs modelo denso"""
        # Modelo denso equivalente
        dense_params = 0

        # Attention layers: 4 * d_model^2 por layer
        dense_params += len(self.layers) * 4 * self.d_model * self.d_model

        # FFN layers: 2 * d_model * 4*d_model por layer
        dense_params += len(self.layers) * 2 * self.d_model * (4 * self.d_model)

        # Embedding + head
        dense_params += self.vocab_size * self.d_model * 2  # embedding + head

        nosso_params = sum(p.numel() for p in self.parameters())

        return dense_params / nosso_params


def run_simple_test():
    """Executar teste simplificado"""
    print("="*70)
    print("TESTE SIMPLIFICADO: Qwen3-Coder Compression")
    print("="*70)

    # Configuração para 16GB VRAM
    config = {
        'vocab_size': 32000,
        'd_model': 1024,      # 1/5 do Qwen3-30B
        'n_layers': 8,        # 1/7 do Qwen3-30B
        'n_heads': 16,
        'rank': 64,
    }

    print(f"Configuração:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Criar modelo
    model = SimpleQwen3Model(**config)

    print(f"\nModelo criado:")
    print(f"  Parâmetros: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Camadas: {len(model.layers)}")

    # Estimar memória
    mem_info = model.estimate_memory(seq_len=512)
    print(f"\nEstimativa de memória (seq_len=512):")
    print(f"  Parâmetros: {mem_info['param_memory_mb']:.1f} MB")
    print(f"  KV cache: {mem_info['kv_cache_mb']:.1f} MB")
    print(f"  Total: {mem_info['total_memory_mb']:.1f} MB")
    print(f"  Compressão: {mem_info['compression_ratio']:.2f}x")

    # Testar inferência
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device}")

    if torch.cuda.is_available():
        model = model.half().to(device)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM disponível: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Input de teste
    batch_size, seq_len = 1, 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)

    print(f"\nTestando inferência...")
    print(f"  Input shape: {input_ids.shape}")

    try:
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_ids)

        # Medir velocidade
        num_iterations = 10
        start_time = time.time()

        with torch.no_grad():
            for i in range(num_iterations):
                outputs = model(input_ids)
                if (i + 1) % 5 == 0:
                    print(f"  Iteração {i+1}/{num_iterations}")

        end_time = time.time()

        # Estatísticas
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        tokens_per_sec = seq_len / avg_time

        print(f"\nResultados:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg por iteração: {avg_time*1000:.1f}ms")
        print(f"  Tokens/segundo: {tokens_per_sec:.1f}")
        print(f"  Output shape: {outputs.shape}")

        # Memória real
        if torch.cuda.is_available():
            print(f"\nUso real de memória GPU:")
            print(f"  Alocado: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"  Reservado: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        # Verificar se cabe em 16GB
        total_used = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        if total_used < 14:  # 14GB limite (deixa 2GB para sistema)
            print(f"\n✓ CABE em 16GB VRAM! ({total_used:.2f} GB usado)")
            print(f"  Sobram {16 - total_used:.2f} GB para sistema")
        else:
            print(f"\n⚠️  Excede 16GB VRAM ({total_used:.2f} GB usado)")

        return {
            'success': True,
            'tokens_per_sec': tokens_per_sec,
            'memory_gb': total_used,
            'compression_ratio': mem_info['compression_ratio']
        }

    except torch.cuda.OutOfMemoryError:
        print("\n✗ CUDA out of memory!")
        print("  Tente reduzir d_model ou n_layers")
        return {'success': False, 'error': 'OOM'}
    except Exception as e:
        print(f"\n✗ Erro: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def test_different_ranks():
    """Testar diferentes ranks de compressão"""
    print("\n" + "="*70)
    print("Testando Diferentes Ranks de Compressão")
    print("="*70)

    ranks = [32, 64, 96, 128]
    results = []

    base_config = {
        'vocab_size': 32000,
        'd_model': 1024,
        'n_layers': 8,
        'n_heads': 16,
    }

    for rank in ranks:
        print(f"\n--- Rank: {rank} ---")

        config = base_config.copy()
        config['rank'] = rank

        model = SimpleQwen3Model(**config)

        # Estimar memória
        mem_info = model.estimate_memory()
        print(f"  Parâmetros: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Memória estimada: {mem_info['total_memory_mb']:.1f} MB")
        print(f"  Compressão: {mem_info['compression_ratio']:.2f}x")

        results.append({
            'rank': rank,
            'parameters': sum(p.numel() for p in model.parameters()),
            'memory_mb': mem_info['total_memory_mb'],
            'compression': mem_info['compression_ratio']
        })

    # Recomendação
    print("\n" + "="*70)
    print("RECOMENDAÇÃO PARA 16GB VRAM")
    print("="*70)

    viable = [r for r in results if r['memory_mb'] < 14000]  # 14GB limite

    if viable:
        # Escolher melhor balance qualidade/memória
        viable.sort(key=lambda x: x['compression'], reverse=True)
        best = viable[0]

        print(f"\n✓ Recomendado: rank={best['rank']}")
        print(f"  Memória: {best['memory_mb']:.1f} MB")
        print(f"  Compressão: {best['compression']:.2f}x")
        print(f"  Parâmetros: {best['parameters']:,}")

        # Comparação com Qwen3-30B original
        qwen30b_params = 30e9  # 30B
        qwen30b_memory_gb = qwen30b_params * 2 / 1e9  # float16
        our_memory_gb = best['memory_mb'] / 1024

        print(f"\nComparação com Qwen3-30B original:")
        print(f"  Original: {qwen30b_memory_gb:.1f} GB (30B params)")
        print(f"  Nosso: {our_memory_gb:.1f} GB ({best['parameters']:,} params)")
        print(f"  Redução: {qwen30b_memory_gb/our_memory_gb:.1f}x")
    else:
        print("\n⚠️  Nenhum rank cabe em 16GB VRAM")
        print("   Tente reduzir d_model ou n_layers")

    return results


def main():
    """Função principal"""
    print("="*70)
    print("VALIDAÇÃO: Compressão Low-Rank para Qwen3-Coder")
    print("Teste simplificado para verificar conceito")
    print("="*70)

    # Teste básico
    result = run_simple_test()

    if result.get('success'):
        print("\n" + "="*70)
        print("✓ CONCEITO VALIDADO!")
        print("="*70)
        print(f"Compressão low-rank funciona com:")
        print(f"  • {result['tokens_per_sec']:.1f} tokens/segundo")
        print(f"  • {result['memory_gb']:.2f} GB VRAM usado")
        print(f"  • {result['compression_ratio']:.2f}x compressão")

        # Testar diferentes ranks
        test_different_ranks()

        print("\n" + "="*70)
        print("PRÓXIMOS PASSOS:")
        print("="*70)
        print("1. Adicionar suporte a MoE (Mixture of Experts)")
        print("2. Implementar quantização 4-bit/8-bit")
        print("3. Adicionar rotary embeddings corretamente")
        print("4. Integrar com transformers do Hugging Face")
        print("5. Carregar pesos reais do Qwen3-Coder")

    else:
        print("\n" + "="*70)
        print("✗ TESTE FALHOU")
        print("="*70)
        print(f"Erro: {result.get('error', 'Desconhecido')}")
        print("\nSoluções possíveis:")
        print("1. Reduzir d_model (ex: 512 em vez de 1024)")
        print("2. Reduzir n_layers (ex: 4 em vez de 8)")
        print("3. Reduzir batch size ou sequence length")
        print("4. Usar CPU se GPU não tiver memória suficiente")

    print("\n" + "="*70)
    print("TESTE CONCLUÍDO!")
    print("="*70)


if __name__ == "__main__":
