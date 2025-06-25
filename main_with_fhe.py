import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import threading
import time
import copy
from concrete import fhe

# ============ 修正されたFHE暗号化集約システム ============
class FHEModelAggregator:
    """test_fhe_aggregation.pyの成功した技術を使ったモデル重み集約システム"""
    
    def __init__(self, model_structure, num_clients=3, scale_factor=100, max_value=50):
        self.num_clients = num_clients
        self.scale_factor = scale_factor  # 1000 → 10
        self.max_value = max_value        # 新規追加: 20
        
        # モデル構造から重みの形状を取得
        self.weight_shapes = {}
        for name, param in model_structure.named_parameters():
            self.weight_shapes[name] = param.shape
        
        self.circuits = {}
        self._compile_fhe_circuits()
    
    def _compile_fhe_circuits(self):
        """修正されたFHE回路をコンパイル"""
        print("🔒 Compiling FHE circuits for model aggregation...")
        
        for layer_name, shape in self.weight_shapes.items():
            total_elements = np.prod(shape)
            print(f"  Compiling circuit for {layer_name} (shape: {shape}, elements: {total_elements})")
            
            # 修正: test_fhe_aggregation.pyと同じFHE関数
            @fhe.compiler({"weights_matrix": "encrypted"})
            def aggregate_layer_weights(weights_matrix):
                # dtypeパラメータを削除し、浮動小数点除算を使用
                summed = np.sum(weights_matrix, axis=0)
                averaged = summed / weights_matrix.shape[0]  # 浮動小数点除算
                return averaged.astype(np.int32)
            
            # 修正: より安全な範囲の入力セット
            inputset = []
            for _ in range(8):
                sample = np.random.randint(
                    -self.max_value, self.max_value + 1,  # -20から21の範囲
                    size=(self.num_clients, total_elements), 
                    dtype=np.int32
                )
                inputset.append(sample)
            
            try:
                # 修正: configurationの簡素化
                configuration = fhe.Configuration(
                    enable_unsafe_features=True,
                    use_insecure_key_cache=True,
                    insecure_key_cache_location=".keys"
                )
                
                circuit = aggregate_layer_weights.compile(inputset, configuration=configuration)
                circuit.keygen()
                self.circuits[layer_name] = circuit
                print(f"  ✅ Circuit for {layer_name} compiled successfully")
                
            except Exception as e:
                print(f"  ❌ Failed to compile circuit for {layer_name}: {e}")
                self.circuits[layer_name] = None
    
    def encrypt_model_weights(self, model_weights_dict):
        """修正された重み暗号化処理"""
        encrypted_weights = {}
        
        for layer_name, weights_tensor in model_weights_dict.items():
            # PyTorch tensor → NumPy array
            weights_np = weights_tensor.detach().cpu().numpy()
            
            # 修正: より安全な変換処理
            scaled_weights = weights_np * self.scale_factor
            # クリッピングを安全な範囲に
            clipped_weights = np.clip(scaled_weights, -self.max_value, self.max_value)
            int_weights = np.round(clipped_weights).astype(np.int32)
            flattened = int_weights.flatten()
            
            encrypted_weights[layer_name] = flattened
            
        return encrypted_weights
    
    def aggregate_encrypted_models(self, client_weights_list):
        """修正されたFHE集約処理"""
        print("\n🔒 Starting FHE model aggregation...")
        
        # 1. 各クライアントの重みを暗号化形式に変換
        encrypted_weights_list = []
        for i, client_weights in enumerate(client_weights_list):
            print(f"  Encrypting model weights from Client {i+1}...")
            encrypted_weights = self.encrypt_model_weights(client_weights)
            encrypted_weights_list.append(encrypted_weights)
        
        # 2. 各レイヤーごとにFHE集約実行
        aggregated_weights = {}
        
        for layer_name in self.weight_shapes.keys():
            circuit = self.circuits[layer_name]
            
            # 各クライアントの同じレイヤーの重みを収集
            client_layer_weights = np.array([
                client_weights[layer_name] 
                for client_weights in encrypted_weights_list
            ])
            
            if circuit is not None:
                try:
                    print(f"  🔒 FHE aggregating {layer_name}...")
                    
                    # test_fhe_aggregation.pyと同じFHE処理
                    encrypted_input = circuit.encrypt(client_layer_weights)
                    encrypted_result = circuit.run(encrypted_input)
                    decrypted_result = circuit.decrypt(encrypted_result)
                    
                    # 整数 → 浮動小数点変換
                    float_result = decrypted_result.astype(np.float32) / self.scale_factor
                    reshaped = float_result.reshape(self.weight_shapes[layer_name])
                    
                    # NumPy → PyTorch tensor
                    aggregated_weights[layer_name] = torch.from_numpy(reshaped)
                    print(f"  ✅ FHE aggregation completed for {layer_name}")
                    
                except Exception as e:
                    print(f"  ⚠️ FHE failed for {layer_name}, using plaintext fallback: {e}")
                    # フォールバック: 通常の平均計算
                    averaged = np.mean(client_layer_weights, axis=0)
                    float_result = averaged.astype(np.float32) / self.scale_factor
                    reshaped = float_result.reshape(self.weight_shapes[layer_name])
                    aggregated_weights[layer_name] = torch.from_numpy(reshaped)
            else:
                # 平文での平均計算
                print(f"  📝 Plaintext aggregation for {layer_name}...")
                averaged = np.mean(client_layer_weights, axis=0)
                float_result = averaged.astype(np.float32) / self.scale_factor
                reshaped = float_result.reshape(self.weight_shapes[layer_name])
                aggregated_weights[layer_name] = torch.from_numpy(reshaped)
        
        print("✅ FHE model aggregation completed!")
        return aggregated_weights

# ============ 修正されたモデル・データセットクラス ============

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class NeuralNetwork(nn.Module):
    def __init__(self, input_size=3, output_size=2):
        super(NeuralNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        
        # 修正: test_fhe_aggregation.pyと同じ初期値
        with torch.no_grad():
            self.fc.weight.data = torch.tensor([[0.1, 0.2, 0.3],
                                               [0.4, 0.5, 0.6]], dtype=torch.float32)
            self.fc.bias.data = torch.tensor([0.1, 0.2], dtype=torch.float32)

    def forward(self, x):
        return self.fc(x)

class LocalTrainer:
    def __init__(self, model, dataset, optimizer, criterion, device):
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(self, epochs):
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        correct = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            
            for data, target in self.dataset:
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                predicted = torch.argmax(output, dim=1)
                epoch_correct += (predicted == target).sum().item()
                epoch_samples += target.size(0)
                epoch_loss += loss.item()
            
            epoch_accuracy = epoch_correct / epoch_samples * 100
            print(f"    Epoch {epoch + 1}/{epochs}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.2f}%")
            
            total_loss += epoch_loss
            correct += epoch_correct
            total_samples += epoch_samples
        
        overall_accuracy = correct / total_samples * 100
        print(f"  Local training completed: Overall Accuracy = {overall_accuracy:.2f}%")
        return overall_accuracy

    def evaluate(self, test_dataset=None):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        dataset = test_dataset if test_dataset else self.dataset
        
        with torch.no_grad():
            for data, target in dataset:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                predicted = torch.argmax(output, dim=1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
                total_loss += loss.item()
        
        accuracy = correct / total * 100
        avg_loss = total_loss / len(dataset)
        
        print(f"  Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return accuracy, avg_loss

    def get_model_weights(self):
        return copy.deepcopy(self.model.state_dict())

    def set_model_weights(self, weights):
        self.model.load_state_dict(weights)

def create_simple_data(batch_size, num_samples=100, seed=None):
    """簡単な計算用のダミーデータを生成"""
    if seed is not None:
        np.random.seed(seed)
    
    # 3つの入力特徴量（例：x1, x2, x3）
    features = np.random.randn(num_samples, 3).astype(np.float32)
    
    # 簡単なルールでラベルを生成
    # x1 + x2 - x3 > 0 なら クラス1、そうでなければ クラス0
    labels = ((features[:, 0] + features[:, 1] - features[:, 2]) > 0).astype(np.int64)
    
    dataset = CustomDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Client:
    def __init__(self, client_id, data_seed=None):
        self.client_id = client_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = NeuralNetwork(input_size=3, output_size=2)
        self.model.to(self.device)
        
        self.train_loader = create_simple_data(batch_size=16, num_samples=100, seed=data_seed)
        self.test_loader = create_simple_data(batch_size=16, num_samples=50, seed=data_seed+1000 if data_seed else None)
        
        # 修正: 学習率をさらに小さく（より安定した学習）
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        
        self.trainer = LocalTrainer(
            self.model, self.train_loader, self.optimizer, self.criterion, self.device
        )

    def local_update(self, global_weights, epochs=2):
        print(f"Client {self.client_id}: Starting local update")
        
        self.trainer.set_model_weights(global_weights)
        
        print(f"Client {self.client_id}: Pre-training evaluation:")
        pre_accuracy, _ = self.trainer.evaluate(self.test_loader)
        
        print(f"Client {self.client_id}: Starting local training...")
        train_accuracy = self.trainer.train(epochs=epochs)
        
        print(f"Client {self.client_id}: Post-training evaluation:")
        post_accuracy, _ = self.trainer.evaluate(self.test_loader)
        
        print(f"Client {self.client_id} Summary:")
        print(f"  Pre-training accuracy: {pre_accuracy:.2f}%")
        print(f"  Training accuracy: {train_accuracy:.2f}%")
        print(f"  Post-training accuracy: {post_accuracy:.2f}%")
        print(f"  Accuracy improvement: {post_accuracy - pre_accuracy:.2f}%")
        
        return self.trainer.get_model_weights()

# ============ 修正されたFHE対応サーバークラス ============
class FHEServer:
    """FHE暗号化対応サーバー"""
    
    def __init__(self, num_clients=3):
        self.num_clients = num_clients
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # グローバルモデルの初期化
        self.global_model = NeuralNetwork(input_size=3, output_size=2)
        self.global_model.to(self.device)
        
        # 修正: FHE集約システムの初期化パラメータ
        self.fhe_aggregator = FHEModelAggregator(
            self.global_model, 
            num_clients=num_clients,
            scale_factor=100,     
            max_value=50         
        )
        
        # テスト用データローダー
        self.test_loader = create_simple_data(batch_size=32, num_samples=200, seed=9999)
        self.criterion = nn.CrossEntropyLoss()

    def evaluate_global_model(self):
        self.global_model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                loss = self.criterion(output, target)
                
                predicted = torch.argmax(output, dim=1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
                total_loss += loss.item()
        
        accuracy = correct / total * 100
        avg_loss = total_loss / len(self.test_loader)
        
        print(f"Global Model - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return accuracy, avg_loss

    def aggregate_models_with_fhe(self, client_weights_list):
        """🔒 FHE暗号化を使ったモデル集約"""
        print("🔒 FHE-based model aggregation (using test_fhe_aggregation.py technology)...")
        
        # FHE暗号化集約実行
        aggregated_weights = self.fhe_aggregator.aggregate_encrypted_models(client_weights_list)
        
        # グローバルモデルに集約結果を適用
        self.global_model.load_state_dict(aggregated_weights)
        
        print("🔒 FHE Global model updated:")
        for name, param in self.global_model.named_parameters():
            print(f"  {name}: {param.data}")

    def get_global_weights(self):
        return copy.deepcopy(self.global_model.state_dict())

def main():
    print("=== 🔒 Federated Learning with FHE Encryption (Fixed Version) ===")
    
    num_clients = 3
    num_rounds = 3
    
    # FHE対応サーバーを使用
    server = FHEServer(num_clients=num_clients)
    clients = [Client(client_id=i+1, data_seed=i*100) for i in range(num_clients)]
    
    print("\nInitial global model weights:")
    for name, param in server.global_model.named_parameters():
        print(f"  {name}: {param.data}")
    
    print("\nInitial Global Model Evaluation:")
    initial_accuracy, _ = server.evaluate_global_model()
    
    # Federated Learning のメインループ
    for round_num in range(num_rounds):
        print(f"\n{'='*60}")
        print(f"🔒 FEDERATED LEARNING ROUND {round_num + 1} (FHE ENCRYPTED)")
        print(f"{'='*60}")
        
        global_weights = server.get_global_weights()
        
        client_weights_list = []
        for client in clients:
            print(f"\n--- Client {client.client_id} Local Update ---")
            local_weights = client.local_update(global_weights, epochs=2)
            client_weights_list.append(local_weights)
        
        # 🔒 FHE暗号化集約（修正版）
        print(f"\n--- 🔒 FHE Server Aggregation ---")
        server.aggregate_models_with_fhe(client_weights_list)
        
        print(f"\nRound {round_num + 1} Global Model Evaluation:")
        current_accuracy, _ = server.evaluate_global_model()
        
        print(f"🔒 Encrypted Round {round_num + 1} completed!")
    
    print(f"\n{'='*60}")
    print("🔒 ENCRYPTED FEDERATED LEARNING COMPLETED!")
    print(f"{'='*60}")
    
    print("\nFinal global model weights:")
    for name, param in server.global_model.named_parameters():
        print(f"  {name}: {param.data}")
    
    print("\nFinal Global Model Evaluation:")
    final_accuracy, _ = server.evaluate_global_model()
    print(f"Overall improvement: {final_accuracy - initial_accuracy:.2f}%")

if __name__ == "__main__":
    main()