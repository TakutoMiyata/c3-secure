import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from concrete import fhe

# ============ FHE暗号化用のユーティリティ ============
class FHEAggregator:
    """FHEを使った重み集約クラス"""
    
    def __init__(self, weight_shapes, scale_factor=1000):
        self.weight_shapes = weight_shapes
        self.scale_factor = scale_factor  # 浮動小数点を整数に変換するスケール
        self.circuits = {}
        self._compile_circuits()
    
    def _compile_circuits(self):
        """各レイヤーごとにFHE回路をコンパイル"""
        print("Compiling FHE circuits for weight aggregation...")
        
        for layer_name, shape in self.weight_shapes.items():
            # 重みの形状に応じて回路を作成
            total_elements = np.prod(shape)
            
            @fhe.compiler({"weights_list": "encrypted"})
            def aggregate_weights(weights_list):
                # weights_list: [num_clients, total_elements]
                summed = np.sum(weights_list, axis=0)
                return summed // weights_list.shape[0]  # 平均計算
            
            # 代表入力を作成（3クライアント想定）
            inputset = [
                np.random.randint(-self.scale_factor, self.scale_factor, 
                                size=(3, total_elements), dtype=np.int64)
                for _ in range(5)
            ]
            
            circuit = aggregate_weights.compile(inputset, composable=True)
            circuit.keygen()
            self.circuits[layer_name] = circuit
            print(f"  Circuit for {layer_name} (shape: {shape}) compiled")
    
    def encrypt_weights(self, weights_dict):
        """クライアントの重みを暗号化"""
        encrypted_weights = {}
        
        for layer_name, weights in weights_dict.items():
            # 浮動小数点を整数に変換
            scaled_weights = (weights * self.scale_factor).astype(np.int64)
            flattened = scaled_weights.flatten()
            
            # 暗号化（実際は平文のまま - デモ用）
            encrypted_weights[layer_name] = flattened
            
        return encrypted_weights
    
    def aggregate_encrypted_weights(self, encrypted_weights_list):
        """暗号化された重みを集約"""
        print("Aggregating encrypted weights using FHE...")
        aggregated_weights = {}
        
        for layer_name in self.weight_shapes.keys():
            # 各クライアントの暗号化重みを収集
            client_weights = np.array([
                client_weights[layer_name] 
                for client_weights in encrypted_weights_list
            ])
            
            # FHE回路で集約実行
            circuit = self.circuits[layer_name]
            encrypted_input = circuit.encrypt(client_weights)
            encrypted_result = circuit.run(encrypted_input)
            decrypted_result = circuit.decrypt(encrypted_result)
            
            # 整数から浮動小数点に戻す
            scaled_back = decrypted_result.astype(np.float32) / self.scale_factor
            reshaped = scaled_back.reshape(self.weight_shapes[layer_name])
            aggregated_weights[layer_name] = reshaped
            
        print("✅ FHE aggregation completed")
        return aggregated_weights

# ============ 既存のクラス（一部修正） ============

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
        
        with torch.no_grad():
            self.fc.weight.fill_(0.1)
            self.fc.bias.fill_(0.0)

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
            for batch_features, batch_labels in self.dataset:
                batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * batch_features.size(0)
                total_samples += batch_features.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == batch_labels).sum().item()
        
        overall_accuracy = correct / total_samples * 100
        print(f"  Local training completed: Overall Accuracy = {overall_accuracy:.2f}%")
        return overall_accuracy

    def get_model_weights(self):
        """モデルの重みをNumPy配列として取得"""
        weights = {}
        for name, param in self.model.named_parameters():
            weights[name] = param.data.cpu().numpy()
        return weights

    def set_model_weights(self, weights):
        """NumPy配列の重みをモデルに設定"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.copy_(torch.from_numpy(weights[name]))

def create_simple_data(batch_size, num_samples=100, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    features = np.random.randn(num_samples, 3).astype(np.float32)
    labels = ((features[:, 0] + features[:, 1] - features[:, 2]) > 0).astype(np.int64)
    
    dataset = CustomDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Client:
    def __init__(self, client_id, data_seed=None):
        self.client_id = client_id
        self.device = torch.device('cpu')
        self.model = NeuralNetwork().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self.data_loader = create_simple_data(batch_size=16, num_samples=100, seed=data_seed)
        self.trainer = LocalTrainer(self.model, self.data_loader, self.optimizer, self.criterion, self.device)

    def local_update(self, global_weights, epochs=2):
        print(f"\n--- Client {self.client_id} Local Training ---")
        self.trainer.set_model_weights(global_weights)
        accuracy = self.trainer.train(epochs)
        updated_weights = self.trainer.get_model_weights()
        return updated_weights, accuracy

class EncryptedServer:
    """FHE対応サーバー"""
    
    def __init__(self, num_clients=3):
        self.device = torch.device('cpu')
        self.global_model = NeuralNetwork().to(self.device)
        self.num_clients = num_clients
        
        # FHE集約器を初期化
        weight_shapes = {name: param.shape for name, param in self.global_model.named_parameters()}
        self.fhe_aggregator = FHEAggregator(weight_shapes)
        
        self.test_data_loader = create_simple_data(batch_size=32, num_samples=200, seed=999)

    def evaluate_global_model(self):
        self.global_model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for features, labels in self.test_data_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.global_model(features)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(self.test_data_loader)
        print(f"Global Model - Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}")
        return accuracy, avg_loss

    def aggregate_models_with_fhe(self, client_weights_list):
        """FHEを使用してモデルを集約"""
        print("\n🔒 Starting FHE-based model aggregation...")
        
        # 1. クライアントの重みを暗号化
        encrypted_weights_list = []
        for i, client_weights in enumerate(client_weights_list):
            print(f"  Encrypting weights from Client {i+1}...")
            encrypted_weights = self.fhe_aggregator.encrypt_weights(client_weights)
            encrypted_weights_list.append(encrypted_weights)
        
        # 2. 暗号化された重みを集約
        aggregated_weights = self.fhe_aggregator.aggregate_encrypted_weights(encrypted_weights_list)
        
        # 3. グローバルモデルを更新
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                param.copy_(torch.from_numpy(aggregated_weights[name]))
        
        print("🔒 FHE aggregation completed successfully!")

    def get_global_weights(self):
        weights = {}
        for name, param in self.global_model.named_parameters():
            weights[name] = param.data.cpu().numpy()
        return weights

def main():
    print("=== 🔒 Encrypted Federated Learning with FHE ===")
    
    num_clients = 3
    num_rounds = 3
    
    # FHE対応サーバーを初期化
    server = EncryptedServer(num_clients=num_clients)
    clients = [Client(client_id=i+1, data_seed=i*100) for i in range(num_clients)]
    
    print("\nInitial global model weights:")
    for name, param in server.global_model.named_parameters():
        print(f"  {name}: {param.data}")
    
    print("\nInitial Global Model Evaluation:")
    initial_accuracy, _ = server.evaluate_global_model()
    
    # Federated Learning のメインループ
    for round_num in range(num_rounds):
        print(f"\n{'='*60}")
        print(f"🔄 FEDERATED LEARNING ROUND {round_num + 1}/{num_rounds}")
        print(f"{'='*60}")
        
        # グローバル重みを取得
        global_weights = server.get_global_weights()
        
        # 各クライアントでローカル学習
        client_weights_list = []
        client_accuracies = []
        
        for client in clients:
            updated_weights, accuracy = client.local_update(global_weights, epochs=2)
            client_weights_list.append(updated_weights)
            client_accuracies.append(accuracy)
        
        print(f"\nClient accuracies: {[f'{acc:.2f}%' for acc in client_accuracies]}")
        
        # 🔒 FHEを使用してモデル集約
        server.aggregate_models_with_fhe(client_weights_list)
        
        # グローバルモデル評価
        print(f"\nGlobal Model Evaluation after Round {round_num + 1}:")
        server.evaluate_global_model()
    
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