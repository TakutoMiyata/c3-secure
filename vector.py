import numpy as np
from concrete import fhe

# ---------------- パラメータ ----------------
n, m = 4, 5
# n個の重み，m人のクライアント
bit_max = 32          # 入力値上限（0〜31）
# -------------------------------------------

# ---------------- FHE 回路 ------------------
@fhe.compiler({"vecs": "encrypted"})
def mean_vector(vecs):
    summed = np.sum(vecs, axis=0)    # shape = (m,)
    return summed // n               # 整数平均
# -------------------------------------------

# 代表入力を複数用意して回路をコンパイル
inputset = [
    np.random.randint(0, bit_max, size=(n, m), dtype=np.int64)
    for _ in range(10)
]

print("Compiling…")
circuit = mean_vector.compile(inputset, composable=True)
print("Key-gen…")
circuit.keygen()

# ---------- 実データ生成 & 暗号化 ----------
plain = np.random.randint(0, bit_max, size=(n, m), dtype=np.int64)
enc   = circuit.encrypt(plain)
mean_enc = circuit.run(enc)
mean_dec = circuit.decrypt(mean_enc)

# ---------- 平文側の計算 -------------------
plain_mean = plain.sum(axis=0) // n

# ---------- 検証 ---------------------------
print("plain vectors:\n", plain)
print("plain mean   :", plain_mean)
print("FHE mean     :", mean_dec)

assert np.array_equal(plain_mean, mean_dec), "❌ 一致しません！"
print("✅ 平文計算と暗号計算の結果は完全に一致しました。")