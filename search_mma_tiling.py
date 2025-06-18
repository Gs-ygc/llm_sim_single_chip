# up proj	1	64	7168	4096
# down proj	1	64	2048	7168
# up proj	2	512	7168	4096
# down proj	2	512	2048	7168
# q_up	1	64	128	24576
# Q @ W_{UV}^T	128	64	128	512
# kv_nope_mm_W_KV_down	1	64	7168	576
# qK	32	256	576	8192
# PV	32	256	8192	512
# O @ W_{UV}	128	64	512	128
# O proj	1	64	16384	7168

class MMAConfig:
    def __init__(self, name, bs, m, k, n):
        self.name = name
        self.bs = bs
        self.m = m
        self.k = k
        self.n = n

    def __str__(self):
        return f"{self.name}\t bs={self.bs}\t m={self.m}\t k={self.k}\t n={self.n}"

configs = [
    MMAConfig("up proj shared", 1, 64, 7168, 4096),
    MMAConfig("down proj shared", 1, 64, 2048, 7168),
    MMAConfig("up proj routed", 2, 512, 7168, 4096),
    MMAConfig("down proj route", 2, 512, 2048, 7168),
    MMAConfig("q_up", 1, 64, 128, 24576),
    MMAConfig("Q @ W_{UV}^T", 128, 64, 128, 512),
    MMAConfig("kv_nope_mm_W_KV_down", 1, 64, 7168, 576),
    MMAConfig("qK", 32, 256, 576, 8192),
    MMAConfig("PV", 32, 256, 8192, 512),
    MMAConfig("O @ W_{UV}", 128, 64, 512, 128),
    MMAConfig("O proj", 1, 64, 16384, 7168),
]

SRAM_size = 2*1024*1024

tile_quants = (64, 256, 64)
