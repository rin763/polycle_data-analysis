# pip install -q mecab-python3 unidic-lite
# pip install -q scikit-learn pandas numpy fugashi ipadic matplotlib seaborn

import pandas as pd
import numpy as np
import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib
matplotlib.use('Agg')  # GUIを使わないバックエンドを設定
import matplotlib.pyplot as plt
import seaborn as sns

# MeCabのTaggerを初期化
try:
    # 辞書が正しくインストールされているか確認
    tagger = MeCab.Tagger()
except Exception as e:
    # エラー時の代替初期化
    tagger = MeCab.Tagger("-r /etc/mecabrc -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")

# グラフの日本語表示設定
import matplotlib.font_manager as fm

# 利用可能な日本語フォントを検索
japanese_fonts = ['Hiragino Kaku Gothic ProN', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao Gothic']
available_fonts = [f.name for f in fm.fontManager.ttflist]

japanese_font = None
for font in japanese_fonts:
    if font in available_fonts:
        japanese_font = font
        break

if japanese_font:
    plt.rcParams['font.family'] = japanese_font
    print(f"✅ 日本語フォント '{japanese_font}' を設定しました。")
else:
    print("⚠️ 日本語フォントが見つかりません。デフォルトフォントを使用します。")
    # フォールバック: システムのデフォルトフォントを使用
    plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['axes.unicode_minus'] = False

file_name = "/Users/rin/Documents/Polycle/polycle_data-analysis/PXイベント参加者DB.csv"

# DataFrameへの読み込み
try:
    df = pd.read_csv(file_name, encoding='utf-8')
except UnicodeDecodeError:
    print("UTF-8で失敗。Shift-JISで再試行します。")
    df = pd.read_csv(file_name, encoding='shift_jis')

print(f"✅ ファイル '{file_name}' の読み込みが完了しました。")
print(df.head(2))

# '学年'→数値型
df['学年'] = pd.to_numeric(df['学年'].astype(str).str.replace(r'[^\d]', '', regex=True), errors='coerce')

# イベント内容TF-IDFベクトル化

def tokenize_text(text, tagger_instance):
    """MeCabで単語分割し、意味のある名詞、動詞、形容詞のみを抽出"""
    if pd.isna(text) or text is None: return ""
    node = tagger_instance.parseToNode(str(text))
    keywords = []

    while node:
        feature = node.feature.split(',')
        # 安全チェック: リストの長さが7未満の場合は原形[6]がないためスキップ
        if len(feature) >= 7 and feature[0] in ['名詞', '動詞', '形容詞']:
            base = feature[6] if feature[6] != '*' else node.surface
            if base not in ['こと', 'もの', 'それ', 'する', 'なる', 'ある', 'いる', '的']:
                keywords.append(base)
        node = node.next
    return " ".join(keywords)

# token化
df['イベント内容_tokens'] = df['イベント内容'].apply(lambda x: tokenize_text(x, tagger))

# TF-IDFベクトルの計算
tfidf = TfidfVectorizer(min_df=2)
tfidf_matrix = tfidf.fit_transform(df['イベント内容_tokens']).toarray()
tfidf_cols = [f'tfidf_{w}' for w in tfidf.get_feature_names_out()]
tfidf_df = pd.DataFrame(tfidf_matrix, columns=tfidf_cols, index=df.index)

# 元のDataFrameと結合
df = pd.concat([df, tfidf_df], axis=1)

# --- 3. 時間帯の特徴量（週末フラグ）---
df['週末フラグ'] = pd.to_datetime(df['日にち'], errors='coerce').dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

print("TF-IDFベクトルがデータに追加されました。")

def categorize_major(major_name):
    """
    学科名を指定された11のカテゴリに分類する関数
    """
    if pd.isna(major_name):
        return 'その他'

    name = str(major_name).upper()

    if '情報' in name or '資訊' in name or '系統' in name or 'CS' in name or '資' in name:
        return '情報'
    if '医学' in name or '醫' in name or '薬学' in name or '營養' in name:
        return '医学'
    if '工学' in name or '工程' in name or '電機' in name or '機械' in name or 'システム' in name or '工學' in name or '半導体' in name:
        return '工学'
    if '理学' in name or '科學' in name or '物理' in name or '數學' in name or '化學' in name:
        return '理学'
    if '建築' in name:
        return '建築'

    if '貿易' in name or '國際' in name or '國' in name or '国際' in name or '外交' in name or 'Global' in name:
        return '国際'
    if '言語' in name or '語學' in name or '日文' in name or '英文' in name or '英語' in name or '外文' in name or '中文' in name or 'BIBA' in name or '文學' in name in name or '人文' in name in name or '社会学' in name or 'コミュニケーション' in name or '傳播' in name or '新聞' in name or '廣播' in name or '文学' in name:
        return '人文'
    if '経済' in name or '企管' in name or '商學' in name or 'Business' in name or '經濟' in name or '経営' in name or '管理' in name or '金融' in name or '會計' in name or 'マーケティング' in name or '行銷' in name or '金' in name or '政治' in name or 'Management' in name:
        return '商学'
    if '教育' in name or '師資' in name or '教育' in name or '教養' in name:
        return '教育'
    if '観光' in name or '觀光' in name:
        return '観光'
    if '運動' in name:
        return '運動'
    if '設計' in name:
        return 'デザイン'

    return 'その他'

df['学科カテゴリー'] = df['学科'].apply(categorize_major)

print("✅ 学科名のカテゴリー分類が完了しました。")
print(df['学科カテゴリー'].value_counts())
print(df['学科カテゴリー'])

other_majors = df[df['学科カテゴリー'] == 'その他']['学科'].unique()

print("\n--- その他 ---")
if len(other_majors) > 0:
    for major in other_majors:
        print(f"- {major}")


# 日本語カテゴリから英語カテゴリへのマッピング辞書（matlibで豆腐回避するため）
category_mapping = {
    '情報': 'IT',
    '商学': 'Business',
    '人文': 'Humanities',
    '国際': 'International',
    '理学': 'Science',
    '教育': 'Education',
    '工学': 'Engineering',
    '医学': 'Medicine',
    '建築': 'Architecture',
    'デザイン': 'Design',
    '運動': 'Exercise',
    '観光': 'Tourism',
    'その他': 'Other'
}

# 新しい英語カテゴリ列を作成
df['Major_Category_EN'] = df['学科カテゴリー'].map(category_mapping).fillna('Other')

# マッピングが漏れた場合（理論的にはないはず）に備えて確認
df.loc[~df['Major_Category_EN'].isin(category_mapping.values()), 'Major_Category_EN'] = 'Other'

print("✅ 新しい列 'Major_Category_EN' が作成されました。")
print(df[['学科カテゴリー', 'Major_Category_EN']].head())

import matplotlib.pyplot as plt
import seaborn as sns

analysis_target = 'Major_Category_EN'


manual_keywords_japanese = ['シュウカツ', 'オオニンズウ', 'コウリュウ', 'ワイワイ', 'ジョウホウ','OBOG', 'オンライン']


# 自動抽出された上位10単語を取得
tfidf_cols = [col for col in df.columns if col.startswith('tfidf_')]
top_auto_keywords = [
    col.replace('tfidf_', '')
    for col in df[tfidf_cols].sum().sort_values(ascending=False).head(10).index.tolist()
]

# 3. 自動抽出単語と手動単語を統合
all_target_words = list(set(top_auto_keywords + manual_keywords_japanese))

all_target_words = [
    f'tfidf_{word}' for word in all_target_words
    if f'tfidf_{word}' in df.columns
]

print("--- 💡 フィルタリング後の最終キーワード ---")
print(f"使用するキーワード: {len(all_target_words)} 個")
print(all_target_words)


# 1. 【修正】集計に使用するTF-IDF列名の定義
keywords_to_use_for_agg = [col for col in all_target_words if col in df.columns]

if not keywords_to_use_for_agg:
    print("Error: None of the specified keywords were found as columns in the DataFrame. Stopping aggregation.")

    pivot_data_en = pd.DataFrame()
else:
    print(f"✅ 集計に使用するTF-IDF列: {keywords_to_use_for_agg}")



analysis_target = 'Major_Category_EN'

# 1. データ集計と前処理

# Calculate TF-IDF means separately
tfidf_agg = df.groupby(analysis_target)[all_target_words].mean()

# Calculate mail counts separately
if 'メールアドレス' in df.columns:
    count_agg = df.groupby(analysis_target)['メールアドレス'].size().to_frame(name='mail_count')
else:
    print("Error: 'メールアドレス' column not found. Cannot calculate mail counts.")
    count_agg = pd.DataFrame(index=df[analysis_target].unique(), columns=['mail_count'])

# Merge the two aggregated DataFrames
pivot_data_en = tfidf_agg.join(count_agg, how='left')

# TF-IDFのNaN（無関心）を0に置換
pivot_data_en = pivot_data_en.fillna(0)

# 参加者3名以上のカテゴリーに絞る
if 'mail_count' in pivot_data_en.columns:
    pivot_data_en = pivot_data_en[pivot_data_en['mail_count'] > 2]
    pivot_data_en = pivot_data_en.drop(columns=['mail_count']).reset_index()
else:
    print("Warning: 'mail_count' column not available for filtering. Proceeding without filtering.")
    pivot_data_en = pivot_data_en.reset_index()


# 2. 列名のリネーム（番号に置き換え）
# ヒートマップに使用される最終的な列名（TF-IDF列）のリストを取得
tfidf_column_names = [col for col in pivot_data_en.columns if col.startswith('tfidf_')]
num_keywords = len(tfidf_column_names)

number_labels = [str(i) for i in range(num_keywords)]
rename_map_numbers = dict(zip(tfidf_column_names, number_labels))

pivot_data_top_final = pivot_data_en.rename(columns=rename_map_numbers)

print("✅ データ集計と番号へのリネームが完了しました。")


# 3. ヒートマップの可視化

plt.figure(figsize=(16, max(8, len(pivot_data_top_final) * 0.8)))

if analysis_target in pivot_data_top_final.columns and not pivot_data_top_final.empty:
    heatmap_data = pivot_data_top_final.set_index(analysis_target)

    final_number_columns = number_labels
    heatmap_data = heatmap_data[final_number_columns]

    # データを転置して縦軸と横軸を逆にする
    heatmap_data_transposed = heatmap_data.T
    
    sns.heatmap(
        heatmap_data_transposed,
        cmap='YlGnBu',
        annot=True,
        fmt=".3f",
        linewidths=.5,
        linecolor='lightgray',
        annot_kws={"fontsize": 10}
    )

    plt.title('Major Category Event Interest Heatmap (Keyword Index)')
    plt.ylabel('Keyword Index (0, 1, 2, 3, ...)')
    plt.xlabel('Major Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # グラフをファイルに保存
    output_filename = 'major/major_event_heatmap.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ ヒートマップを '{output_filename}' に保存しました。")
    
    plt.close()

else:
    print("Warning: Final data for heatmap is empty or missing the index column. Cannot plot.")



# キーワード対応表の出力
print("\n--- 💡 キーワード対応表 ---")
japanese_keywords = [kw.replace('tfidf_', '') for kw in all_target_words]
keyword_table = pd.DataFrame({
    'Index': number_labels,
    'Keyword (Japanese)': japanese_keywords
})
print(keyword_table)

# キーワード対応表をファイルに保存
keyword_table.to_csv('major/major_event_heatmap_keywords.txt', sep='\t', index=False)
print("✅ キーワード対応表を 'major_heatmap_keywords.txt' に保存しました。")