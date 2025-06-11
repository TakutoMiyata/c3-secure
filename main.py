import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import threading
import time
import copy

# データセットクラス
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ニューラルネットワークモデル
class NeuralNetwork(nn.Module):
    def __init__(self, input_size=3, output_size=2):
        super(NeuralNetwork, self).__init__()
        # 入力層から出力層への直接接続（隠れ層なし）
        self.fc = nn.Linear(input_size, output_size)
        
        # 重みを手動で設定
        with torch.no_grad():
            self.fc.weight.data = torch.tensor([[1.5, -0.8, 2.1],
                                               [-1.2, 2.3, 0.7]], dtype=torch.float32)
            self.fc.bias.data = torch.tensor([0.5, -0.3], dtype=torch.float32)

    def forward(self, x):
        return self.fc(x)

# ローカルトレーナークラス
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
                
                # 精度計算
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
        """モデルの評価を行う"""
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

# データ生成関数
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

# クライアントクラス
class Client:
    def __init__(self, client_id, data_seed=None):
        self.client_id = client_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # モデルの初期化
        self.model = NeuralNetwork(input_size=3, output_size=2)
        self.model.to(self.device)
        
        # クライアント固有のデータを生成（異なるseedで）
        self.train_loader = create_simple_data(batch_size=16, num_samples=100, seed=data_seed)
        self.test_loader = create_simple_data(batch_size=16, num_samples=50, seed=data_seed+1000 if data_seed else None)
        
        # オプティマイザーと損失関数
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        self.criterion = nn.CrossEntropyLoss()
        
        # ローカルトレーナー
        self.trainer = LocalTrainer(
            self.model, self.train_loader, self.optimizer, self.criterion, self.device
        )

    def local_update(self, global_weights, epochs=2):
        """ローカル更新を実行"""
        print(f"Client {self.client_id}: Starting local update")
        
        # グローバル重みを設定
        self.trainer.set_model_weights(global_weights)
        
        # ラウンド開始前の精度評価
        print(f"Client {self.client_id}: Pre-training evaluation:")
        pre_accuracy, _ = self.trainer.evaluate(self.test_loader)
        
        # ローカル学習
        print(f"Client {self.client_id}: Starting local training...")
        train_accuracy = self.trainer.train(epochs=epochs)
        
        # ラウンド終了後の精度評価
        print(f"Client {self.client_id}: Post-training evaluation:")
        post_accuracy, _ = self.trainer.evaluate(self.test_loader)
        
        print(f"Client {self.client_id} Summary:")
        print(f"  Pre-training accuracy: {pre_accuracy:.2f}%")
        print(f"  Training accuracy: {train_accuracy:.2f}%")
        print(f"  Post-training accuracy: {post_accuracy:.2f}%")
        print(f"  Accuracy improvement: {post_accuracy - pre_accuracy:.2f}%")
        
        # 更新された重みを返す
        return self.trainer.get_model_weights()

# サーバークラス
class Server:
    def __init__(self, num_clients=3):
        self.num_clients = num_clients
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # グローバルモデルの初期化
        self.global_model = NeuralNetwork(input_size=3, output_size=2)
        self.global_model.to(self.device)
        
        # テスト用データローダー
        self.test_loader = create_simple_data(batch_size=32, num_samples=200, seed=9999)
        self.criterion = nn.CrossEntropyLoss()

    def evaluate_global_model(self):
        """グローバルモデルの精度を評価"""
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

    def aggregate_models(self, client_weights_list):
        """FedAvg: クライアントのモデル重みを平均化"""
        print("Aggregating models using FedAvg...")
        
        # 各レイヤーの重みを平均化
        global_weights = self.global_model.state_dict()
        
        for key in global_weights.keys():
            global_weights[key] = torch.stack([
                client_weights[key] for client_weights in client_weights_list
            ]).mean(0)
            
        self.global_model.load_state_dict(global_weights)
        
        print("Global model updated:")
        for name, param in self.global_model.named_parameters():
            print(f"  {name}: {param.data}")

    def get_global_weights(self):
        return copy.deepcopy(self.global_model.state_dict())

# メイン関数
def main():
    print("=== Federated Learning (Single File Implementation) ===")
    
    # サーバーとクライアントを初期化
    num_clients = 3
    num_rounds = 3
    
    server = Server(num_clients=num_clients)
    clients = [Client(client_id=i+1, data_seed=i*100) for i in range(num_clients)]
    
    print("\nInitial global model weights:")
    for name, param in server.global_model.named_parameters():
        print(f"  {name}: {param.data}")
    
    # 初期グローバルモデルの精度評価
    print("\nInitial Global Model Evaluation:")
    initial_accuracy, _ = server.evaluate_global_model()
    
    # Federated Learning のメインループ
    for round_num in range(num_rounds):
        print(f"\n{'='*50}")
        print(f"FEDERATED LEARNING ROUND {round_num + 1}")
        print(f"{'='*50}")
        
        # グローバル重みを取得
        global_weights = server.get_global_weights()
        
        # 各クライアントでローカル更新
        client_weights_list = []
        for client in clients:
            print(f"\n--- Client {client.client_id} Local Update ---")
            local_weights = client.local_update(global_weights, epochs=2)
            client_weights_list.append(local_weights)
        
        # サーバーでモデルを集約
        print(f"\n--- Server Aggregation ---")
        server.aggregate_models(client_weights_list)
        
        # グローバルモデルの精度評価
        print(f"\nRound {round_num + 1} Global Model Evaluation:")
        current_accuracy, _ = server.evaluate_global_model()
        
        print(f"Round {round_num + 1} completed!")
    
    print(f"\n{'='*50}")
    print("FEDERATED LEARNING COMPLETED!")
    print(f"{'='*50}")
    
    print("\nFinal global model weights:")
    for name, param in server.global_model.named_parameters():
        print(f"  {name}: {param.data}")
    
    # 最終グローバルモデルの精度評価
    print("\nFinal Global Model Evaluation:")
    final_accuracy, _ = server.evaluate_global_model()
    print(f"Overall improvement: {final_accuracy - initial_accuracy:.2f}%")

if __name__ == "__main__":
    main()
