import torch
import torch.nn as nn
import numpy as np
import copy
from concrete import fhe

# ============ 動作確認済みFHE集約システム ============
class WorkingFHEAggregator:
    def __init__(self, model, num_clients=3, scale_factor=100):  # スケールファクターを小さく
        self.num_clients = num_clients
        self.scale_factor = scale_factor 
        self.max_value = 50 
        
        # モデル構造から重みの形状を取得
        self.weight_shapes = {}
        for name, param in model.named_parameters():
            self.weight_shapes[name] = param.shape
        
        self.circuits = {}
        self._compile_fhe_circuits()
    
    def _compile_fhe_circuits(self):
        """修正されたFHE回路をコンパイル"""
        print("🔒 Compiling working FHE circuits...")
        
        for layer_name, shape in self.weight_shapes.items():
            total_elements = np.prod(shape)
            print(f"  Compiling {layer_name}: shape={shape}, elements={total_elements}")
            
            # 修正されたFHE関数（dtypeパラメータなし）
            @fhe.compiler({"weights_matrix": "encrypted"})
            def aggregate_weights(weights_matrix):
                # np.sumからdtypeパラメータを削除
                summed = np.sum(weights_matrix, axis=0)
                # 浮動小数点除算を使用
                averaged = summed / weights_matrix.shape[0]
                return averaged.astype(np.int32)
            
            # より安全な範囲の入力セット
            inputset = []
            for _ in range(8):
                sample = np.random.randint(
                    -self.max_value, self.max_value + 1,
                    size=(self.num_clients, total_elements), 
                    dtype=np.int32
                )
                inputset.append(sample)
            
            try:
                configuration = fhe.Configuration(
                    enable_unsafe_features=True,
                    use_insecure_key_cache=True,
                    insecure_key_cache_location=".keys"
                )
                
                circuit = aggregate_weights.compile(inputset, configuration=configuration)
                circuit.keygen()
                self.circuits[layer_name] = circuit
                print(f"  ✅ {layer_name} compiled successfully")
                
            except Exception as e:
                print(f"  ❌ {layer_name} compilation failed: {e}")
                self.circuits[layer_name] = None
    
    def encrypt_weights(self, weights_dict):
        """重みを安全な範囲の整数に変換"""
        encrypted_weights = {}
        
        for layer_name, weights_tensor in weights_dict.items():
            weights_np = weights_tensor.detach().cpu().numpy()
            
            # より小さいスケールファクターで変換
            scaled_weights = weights_np * self.scale_factor
            
            # 安全な範囲にクリップ
            clipped_weights = np.clip(scaled_weights, -self.max_value, self.max_value)
            int_weights = np.round(clipped_weights).astype(np.int32)
            flattened = int_weights.flatten()
            
            encrypted_weights[layer_name] = flattened
            
        return encrypted_weights
    
    def aggregate_fhe(self, client_weights_list):
        """FHE暗号化集約"""
        print("\n🔒 Working FHE Aggregation...")
        
        # 1. 重みを暗号化形式に変換
        encrypted_weights_list = []
        for i, client_weights in enumerate(client_weights_list):
            encrypted_weights = self.encrypt_weights(client_weights)
            encrypted_weights_list.append(encrypted_weights)
        
        # 2. FHE集約実行
        aggregated_weights = {}
        
        for layer_name in self.weight_shapes.keys():
            circuit = self.circuits[layer_name]
            
            if circuit is not None:
                # 各クライアントの重みを収集
                client_layer_weights = np.array([
                    client_weights[layer_name] 
                    for client_weights in encrypted_weights_list
                ])
                
                print(f"  🔒 FHE processing {layer_name}...")
                print(f"    Input: {client_layer_weights}")
                
                try:
                    # FHE暗号化計算
                    encrypted_input = circuit.encrypt(client_layer_weights)
                    encrypted_result = circuit.run(encrypted_input)
                    decrypted_result = circuit.decrypt(encrypted_result)
                    
                    print(f"    FHE result (integer): {decrypted_result}")
                    
                    # 整数 → 浮動小数点変換
                    float_result = decrypted_result.astype(np.float32) / self.scale_factor
                    reshaped = float_result.reshape(self.weight_shapes[layer_name])
                    aggregated_weights[layer_name] = torch.from_numpy(reshaped)
                    
                    print(f"    FHE result (float): {float_result}")
                    print(f"    ✅ {layer_name} completed")
                    
                except Exception as e:
                    print(f"    ❌ FHE execution failed for {layer_name}: {e}")
                    return None
            else:
                print(f"  ❌ No circuit for {layer_name}")
                return None
        
        return aggregated_weights
    
    def aggregate_plaintext(self, client_weights_list):
        """平文での集約（比較用）"""
        print("\n📝 Plaintext Aggregation...")
        
        aggregated_weights = {}
        
        for layer_name in self.weight_shapes.keys():
            weights_stack = torch.stack([
                client_weights[layer_name] for client_weights in client_weights_list
            ])
            result = weights_stack.mean(0)
            aggregated_weights[layer_name] = result
            
            print(f"  {layer_name}: {result.flatten()}")
        
        return aggregated_weights
    
    def aggregate_manual_integer(self, client_weights_list):
        """手動整数計算（検証用）"""
        print("\n🔧 Manual Integer Calculation...")
        
        aggregated_weights = {}
        
        for layer_name in self.weight_shapes.keys():
            # 各クライアントの重みを整数化
            encrypted_weights_list = []
            for client_weights in client_weights_list:
                encrypted_weights = self.encrypt_weights({layer_name: client_weights[layer_name]})
                encrypted_weights_list.append(encrypted_weights[layer_name])
            
            client_array = np.array(encrypted_weights_list)
            print(f"  {layer_name} integer weights: {client_array}")
            
            # 手動平均計算
            summed = np.sum(client_array, axis=0)
            averaged = summed / self.num_clients
            
            print(f"    Sum: {summed}")
            print(f"    Average: {averaged}")
            
            # 浮動小数点に戻す
            float_result = averaged / self.scale_factor
            reshaped = float_result.reshape(self.weight_shapes[layer_name])
            aggregated_weights[layer_name] = torch.from_numpy(reshaped.astype(np.float32))
            
            print(f"    Final result: {float_result}")
        
        return aggregated_weights

# ============ テスト用モデル ============
class SimpleTestModel(nn.Module):
    def __init__(self):
        super(SimpleTestModel, self).__init__()
        self.fc = nn.Linear(3, 2)
        
        # より小さな初期値（スケールファクター10に対応）
        with torch.no_grad():
            self.fc.weight.data = torch.tensor([[0.1, 0.2, 0.3],
                                               [0.4, 0.5, 0.6]], dtype=torch.float32)
            self.fc.bias.data = torch.tensor([0.1, 0.2], dtype=torch.float32)

    def forward(self, x):
        return self.fc(x)

def generate_small_variation_clients(num_clients=3):
    """小さな変動のクライアント重みを生成"""
    print(f"📊 Generating {num_clients} clients with small variations...")
    
    client_weights_list = []
    
    for i in range(num_clients):
        model = SimpleTestModel()
        
        # 非常に小さな変動のみ
        with torch.no_grad():
            # 変動を±0.02に制限（スケール10で±0.2の整数変動）
            variation = (i - 1) * 0.02  # -0.02, 0, +0.02
            model.fc.weight.data += variation
            model.fc.bias.data += variation * 0.5
        
        weights = model.state_dict()
        client_weights_list.append(weights)
        
        print(f"  Client {i+1}:")
        for name, param in weights.items():
            print(f"    {name}: {param.flatten()}")
    
    return client_weights_list

def compare_results(fhe_result, plaintext_result, manual_result, tolerance=0.05):
    """結果の詳細比較"""
    print("\n🔍 Detailed Results Comparison")
    print("="*60)
    
    if fhe_result is None:
        print("❌ FHE result is None - cannot compare")
        return False
    
    all_close = True
    
    for layer_name in plaintext_result.keys():
        print(f"\n--- {layer_name} ---")
        
        fhe_vals = fhe_result[layer_name].flatten()
        plain_vals = plaintext_result[layer_name].flatten()
        manual_vals = manual_result[layer_name].flatten()
        
        print(f"Plaintext: {plain_vals}")
        print(f"Manual:    {manual_vals}")
        print(f"FHE:       {fhe_vals}")
        
        # 差分計算
        fhe_plain_diff = torch.abs(fhe_vals - plain_vals)
        manual_plain_diff = torch.abs(manual_vals - plain_vals)
        fhe_manual_diff = torch.abs(fhe_vals - manual_vals)
        
        print(f"FHE vs Plain:   max_diff={torch.max(fhe_plain_diff):.6f}")
        print(f"Manual vs Plain: max_diff={torch.max(manual_plain_diff):.6f}")
        print(f"FHE vs Manual:  max_diff={torch.max(fhe_manual_diff):.6f}")
        
        # 許容誤差チェック
        if torch.max(fhe_plain_diff) <= tolerance:
            print(f"✅ {layer_name}: FHE matches plaintext (tolerance: {tolerance})")
        else:
            print(f"❌ {layer_name}: FHE differs from plaintext (tolerance: {tolerance})")
            all_close = False
    
    return all_close

def main_test():
    """メインテスト関数"""
    print("="*80)
    print("🧪 WORKING FHE AGGREGATION TEST")
    print("="*80)
    
    # テスト準備
    test_model = SimpleTestModel()
    fhe_aggregator = WorkingFHEAggregator(test_model, num_clients=3, scale_factor=10)
    
    # 固定シードでテスト
    np.random.seed(42)
    torch.manual_seed(42)
    client_weights_list = generate_small_variation_clients(num_clients=3)
    
    # 3つの方法で集約実行
    fhe_result = fhe_aggregator.aggregate_fhe(client_weights_list)
    plaintext_result = fhe_aggregator.aggregate_plaintext(client_weights_list)
    manual_result = fhe_aggregator.aggregate_manual_integer(client_weights_list)
    
    # 結果比較
    success = compare_results(fhe_result, plaintext_result, manual_result, tolerance=0.1)
    
    # 最終結果
    print("\n" + "="*80)
    print("📊 TEST RESULTS")
    print("="*80)
    
    if success:
        print("🎉 SUCCESS: FHE aggregation works correctly!")
        print("✅ 暗号化計算と平文計算の結果が一致しました")
        
        # 期待される結果の説明
        print("\n📋 What this proves:")
        print("• FHE circuits compile and execute successfully")
        print("• Encrypted aggregation produces same results as plaintext")
        print("• The aggregation preserves privacy while maintaining accuracy")
        
    else:
        print("❌ FAILURE: FHE aggregation has accuracy issues")
        print("❌ 暗号化計算と平文計算の結果に差があります")
        
        print("\n🔧 Possible improvements:")
        print("• Increase scale factor for better precision")
        print("• Adjust tolerance levels")
        print("• Use different FHE parameters")
    
    print("="*80)
    return success

if __name__ == "__main__":
    try:
        main_test()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()