# -*- coding: utf-8 -*-
"""Minimal benchmark: can 4 blocks run concurrently via CUDA Graphs?"""
import torch
from torch import nn
from pathlib import Path
import statistics

log = Path(r'C:\Users\maxeo\gildnn\runs\graph_concurrency.txt')

with log.open('w', encoding='utf-8') as f:
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    
    class Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.up = nn.Linear(d, 4*d)
            self.act = nn.GELU()
            self.down = nn.Linear(4*d, d)
        def forward(self, x):
            return self.down(self.act(self.up(x)))
    
    def bench_sequential(blocks, x, warmup=20, measure=100):
        bufs = [torch.empty_like(x) for _ in blocks]
        out = torch.empty_like(x)
        
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                for blk, buf in zip(blocks, bufs):
                    buf.copy_(blk(x))
                out.copy_(x)
                for buf in bufs:
                    out.add_(buf)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for blk, buf in zip(blocks, bufs):
                buf.copy_(blk(x))
            out.copy_(x)
            for buf in bufs:
                out.add_(buf)
        torch.cuda.synchronize()
        
        for _ in range(warmup):
            g.replay()
        torch.cuda.synchronize()
        
        samples = []
        for _ in range(measure):
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            g.replay()
            ev1.record()
            ev1.synchronize()
            samples.append(ev0.elapsed_time(ev1))
        
        del g
        return statistics.fmean(samples), statistics.stdev(samples), out.clone()
    
    def bench_parallel(blocks, x, warmup=20, measure=100):
        bufs = [torch.empty_like(x) for _ in blocks]
        out = torch.empty_like(x)
        streams = [torch.cuda.Stream() for _ in blocks]
        
        capture_s = torch.cuda.Stream()
        capture_s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_s):
            for _ in range(3):
                for ws, blk, buf in zip(streams, blocks, bufs):
                    ws.wait_stream(capture_s)
                    with torch.cuda.stream(ws):
                        buf.copy_(blk(x))
                for ws in streams:
                    capture_s.wait_stream(ws)
                out.copy_(x)
                for buf in bufs:
                    out.add_(buf)
        torch.cuda.current_stream().wait_stream(capture_s)
        torch.cuda.synchronize()
        
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=capture_s):
            for ws, blk, buf in zip(streams, blocks, bufs):
                ws.wait_stream(capture_s)
                with torch.cuda.stream(ws):
                    buf.copy_(blk(x))
            for ws in streams:
                capture_s.wait_stream(ws)
            out.copy_(x)
            for buf in bufs:
                out.add_(buf)
        torch.cuda.synchronize()
        
        for _ in range(warmup):
            g.replay()
        torch.cuda.synchronize()
        
        samples = []
        for _ in range(measure):
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            g.replay()
            ev1.record()
            ev1.synchronize()
            samples.append(ev0.elapsed_time(ev1))
        
        del g
        return statistics.fmean(samples), statistics.stdev(samples), out.clone()
    
    header = f"{'shape':<40} {'seq_ms':<16} {'par_ms':<16} {'ratio':<10} {'ok'}"
    f.write(header + '\n')
    f.write('-' * len(header) + '\n')
    f.flush()
    
    configs = [
        (8192, 64, 4),
        (8192, 128, 4),
        (8192, 256, 4),
        (8192, 512, 4),
        (8192, 64, 8),
        (8192, 128, 8),
        (2048, 128, 4),
        (2048, 256, 4),
        (512, 128, 4),
        (512, 256, 4),
    ]
    
    with torch.inference_mode():
        for tokens, d, n_blocks in configs:
            torch.cuda.empty_cache()
            x = torch.randn(tokens, d, device=device)
            blocks = nn.ModuleList([Block(d) for _ in range(n_blocks)]).to(device).eval()
            
            try:
                seq_mean, seq_std, seq_out = bench_sequential(blocks, x)
                par_mean, par_std, par_out = bench_parallel(blocks, x)
                correct = torch.allclose(seq_out, par_out, rtol=1e-5, atol=1e-5)
                ratio = par_mean / seq_mean
                
                label = f'tok={tokens} d={d} blk={n_blocks}'
                line = f'{label:<40} {seq_mean:.3f}+/-{seq_std:.3f}  {par_mean:.3f}+/-{par_std:.3f}  {ratio:.4f}x   {correct}'
                f.write(line + '\n')
            except Exception as e:
                f.write(f'tok={tokens} d={d} blk={n_blocks}: ERROR {e}\n')
            f.flush()
            del blocks, x
    
    f.write('\nDONE. ratio < 1.0 means parallel is faster.\n')
