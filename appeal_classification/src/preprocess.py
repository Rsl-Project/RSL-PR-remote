import pandas as pd
from src.config.dataset_config import DatasetConfig


class Preprocessor:
    '''
    データセットの前処理を行うクラス
    '''
    def __init__(self):
        self.dataset_config = DatasetConfig()
        
    def _merge_similar_documents(self, df_train_raw: pd.DataFrame) -> pd.DataFrame:
        '''
        類似文書をマージする処理

        Args:
            df_train_raw (pd.DataFrame): マージ前の訓練データ
        Returns:
            pd.DataFrame: マージ後の訓練データ
        '''
        merged_dfs = []
        
        print("\n=== 類似文書マージ処理開始 ===")
        print(f"処理前の総レコード数: {len(df_train_raw)}")
        
        for label in range(8):
            class_df = df_train_raw[df_train_raw['label'] == label].copy()
            print(f"\nクラス {label} の処理:")
            print(f"クラス内レコード数: {len(class_df)}")
            
            texts = class_df['title_org'].tolist()
            n = len(texts)
            
            if n <= 1:
                print("レコードが1件以下のため、マージ処理をスキップ")
                merged_dfs.append(class_df)
                continue
            
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(texts)
                
                from sklearn.metrics.pairwise import cosine_similarity
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                print("類似度行列の計算完了")
            except Exception as e:
                print(f"エラー発生: {e}")
                print("エラーにより、このクラスはマージせずにスキップ")
                merged_dfs.append(class_df)
                continue
            
            merged_indices = set()
            merged_groups = []
            similarity_threshold = 0.8
            
            for i in range(n):
                if i in merged_indices:
                    continue
                    
                group = [i]
                for j in range(i+1, n):
                    if j not in merged_indices and similarity_matrix[i][j] >= similarity_threshold:
                        group.append(j)
                        merged_indices.add(j)
                
                if len(group) > 1:
                    merged_groups.append(group)
                else:
                    merged_dfs.append(class_df.iloc[[i]])
            
            for group in merged_groups:
                merged_text = " ".join(class_df.iloc[group]['title_org'])
                merged_row = class_df.iloc[group[0]].copy()
                merged_row['title_org'] = merged_text
                merged_dfs.append(pd.DataFrame([merged_row]))
            
            print(f"マージされたグループ数: {len(merged_groups)}")
            print(f"マージ後のレコード数: {len(merged_groups) + len(class_df) - len(merged_indices)}")
        
        df_train_merged = pd.concat(merged_dfs, ignore_index=True)
        
        print("\n=== マージ処理完了 ===")
        print(f"処理前の総レコード数: {len(df_train_raw)}")
        print(f"処理後の総レコード数: {len(df_train_merged)}")
        print("\nクラスごとのレコード数:")
        print(df_train_merged['label'].value_counts().sort_index())
        
        return df_train_merged

    def _downsample_dataset(self, df_train_raw: pd.DataFrame) -> pd.DataFrame:
        '''
        各クラスのデータ数を最小クラスに合わせてダウンサンプリングする処理

        Args:
            df_train_raw (pd.DataFrame): ダウンサンプリング前の訓練データ
        Returns:
            pd.DataFrame: ダウンサンプリング後の訓練データ
        '''
        # 各クラスのレコード数を取得
        class_counts = df_train_raw['label'].value_counts()
        print("=== Before Downsampling ===")
        print(class_counts)
        print(f"Total records: {len(df_train_raw)}")
        
        # 最小のクラスのレコード数を取得
        min_count = class_counts.min()
        print(f"\nDownsampling to {min_count} records per class")
        
        # 各クラスからmin_count件をランダムに抽出
        balanced_dfs = []
        for label in range(8):
            class_df = df_train_raw[df_train_raw['label'] == label]
            if len(class_df) > min_count:
                balanced_dfs.append(class_df.sample(n=min_count, random_state=42))
            else:
                balanced_dfs.append(class_df)
        
        # 抽出したデータを結合
        df_train_balanced = pd.concat(balanced_dfs)
        
        # シャッフル
        df_train_balanced = df_train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 結果の確認
        print("\n=== After Downsampling ===")
        print(df_train_balanced['label'].value_counts())
        print(f"Total records: {len(df_train_balanced)}")
        
        return df_train_balanced

    def preprocess(self, mode=0, nan_class=True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''
        データセットの前処理を行う.
        実行モードに応じて以下から処理を選択する.
        mode_0: ラベルなしデータを削除.
        mode_1: mode_0 + 類似文書をマージ.
        mode_2: mode_0 + ダウンサンプリング.
        mode_3: mode_0 + 類似文書をマージ + ダウンサンプリング.

        Args:
            input: mode(int)
        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 前処理済みの(訓練データ, 評価データ, テストデータ)
        '''
        # データセット読み込み
        df_train_raw = pd.read_csv(self.dataset_config.train_path)
        df_dev_raw = pd.read_csv(self.dataset_config.dev_path)
        df_test_raw = pd.read_csv(self.dataset_config.test_path)

        if nan_class:
            df_train_raw['appealing_axis'].fillna(8.0, inplace=True)
            df_dev_raw['appealing_axis'].fillna(8.0, inplace=True)
            df_test_raw['appealing_axis'].fillna(8.0, inplace=True)
        else:
            # appealing_axis が null の行を削除
            df_train_raw = df_train_raw.dropna(subset=['appealing_axis'])
            df_dev_raw = df_dev_raw.dropna(subset=['appealing_axis'])
            df_test_raw = df_test_raw.dropna(subset=['appealing_axis'])

        # # appealing_axis列をint型に変換したい
        # print("変換前の型:", df_train_raw['appealing_axis'].dtype)
        # df_train_raw['appealing_axis'] = df_train_raw['appealing_axis'].map(int)
        # print("変換後の型:", df_train_raw['appealing_axis'].dtype)

        # print("変換前の型:", df_dev_raw['appealing_axis'].dtype)
        # df_dev_raw['appealing_axis'] = df_dev_raw['appealing_axis'].map(int)
        # print("変換後の型:", df_dev_raw['appealing_axis'].dtype)

        # print("変換前の型:", df_test_raw['appealing_axis'].dtype)
        # df_test_raw['appealing_axis'] = df_test_raw['appealing_axis'].map(int)
        # print("変換後の型:", df_test_raw['appealing_axis'].dtype)
        
        # 0列目、2列目、4列目を削除
        df_train_raw = df_train_raw.drop(df_train_raw.columns[[0, 2, 4]], axis=1)
        df_dev_raw = df_dev_raw.drop(df_dev_raw.columns[[0, 2, 4]], axis=1)
        df_test_raw = df_test_raw.drop(df_test_raw.columns[[0, 2, 4]], axis=1)

        # title_org 列の欠損値を削除
        df_train_raw = df_train_raw.dropna(subset=['title_org'])
        df_dev_raw = df_dev_raw.dropna(subset=['title_org']) 
        df_test_raw = df_test_raw.dropna(subset=['title_org'])

        # appealing_axis 列を label に変更
        df_train_raw = df_train_raw.rename(columns={'appealing_axis': 'label'})
        df_dev_raw = df_dev_raw.rename(columns={'appealing_axis': 'label'})
        df_test_raw = df_test_raw.rename(columns={'appealing_axis': 'label'})

        if mode == 0:
            return df_train_raw, df_dev_raw, df_test_raw
        
        elif mode == 1:
            df_train_merged = self._merge_similar_documents(df_train_raw)
            df_dev_merged = self._merge_similar_documents(df_dev_raw)
            df_test_merged = self._merge_similar_documents(df_test_raw)
            return df_train_merged, df_dev_merged, df_test_merged

        elif mode == 2:
            df_train_balanced = self._downsample_dataset(df_train_raw)
            df_dev_balanced = self._downsample_dataset(df_dev_raw)
            df_test_balanced = self._downsample_dataset(df_test_raw)
            return df_train_balanced, df_dev_balanced, df_test_balanced

        elif mode == 3:
            df_train_merged = self._merge_similar_documents(df_train_raw)
            df_dev_merged = self._merge_similar_documents(df_dev_raw)
            df_test_merged = self._merge_similar_documents(df_test_raw)
            
            df_train_balanced = self._downsample_dataset(df_train_merged)
            df_dev_balanced = self._downsample_dataset(df_dev_merged)
            df_test_balanced = self._downsample_dataset(df_test_merged)
            return df_train_balanced, df_dev_balanced, df_test_balanced

        else:
            raise ValueError("無効なモードが指定されました。mode は 0-3 の整数を指定してください。")
        

if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.preprocess()