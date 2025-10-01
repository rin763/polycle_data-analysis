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

# '学年'→数値型に変換
df['学年'] = pd.to_numeric(df['学年'].astype(str).str.replace(r'[^\d]', '', regex=True), errors='coerce')

# 学年の分布を確認
print("\n=== 学年の分布 ===")
print(df['学年'].value_counts().sort_index())
print(f"学年の範囲: {df['学年'].min()} - {df['学年'].max()}")

# 学年をカテゴリーに分類
def categorize_grade(grade):
    """学年をカテゴリーに分類"""
    if pd.isna(grade):
        return 'その他'
    
    grade = int(grade)
    
    if grade == 1:
        return '1年生'
    elif grade == 2:
        return '2年生'
    elif grade == 3:
        return '3年生'
    elif grade == 4:
        return '4年生'
    else:
        return '院生/社会人'

df['学年カテゴリー'] = df['学年'].apply(categorize_grade)

print("\n=== 学年カテゴリーの分布 ===")
print(df['学年カテゴリー'].value_counts())

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

print("✅ TF-IDFベクトルがデータに追加されました。")

# 手動で選択したキーワードと自動抽出キーワードの組み合わせ
manual_keywords_japanese = ['シュウカツ', 'オオニンズウ', 'コウリュウ', 'ワイワイ', 'ジョウホウ','OBOG', 'オンライン']

words_to_exclude = ['デキル', 'テイキョウ']

# 自動抽出された上位10単語を取得
tfidf_cols = [col for col in df.columns if col.startswith('tfidf_')]
top_auto_keywords = [
    col.replace('tfidf_', '')
    for col in df[tfidf_cols].sum().sort_values(ascending=False).head(10).index.tolist()
]

# 自動抽出単語と手動単語を統合
all_target_words = list(set(top_auto_keywords + manual_keywords_japanese))

# 除外リストに基づいてキーワードをフィルタリング
final_keywords_to_use = [
    word for word in all_target_words
    if word not in words_to_exclude
]

final_keywords_to_use = [
    f'tfidf_{word}' for word in final_keywords_to_use
    if f'tfidf_{word}' in df.columns
]

print("\n--- 💡 フィルタリング後の最終キーワード ---")
print(f"使用するキーワード: {len(final_keywords_to_use)} 個")
print(final_keywords_to_use)

# 学年別のTF-IDF分析
analysis_target = '学年カテゴリー'

# 集計に使用するTF-IDF列名の定義
keywords_to_use_for_agg = [col for col in final_keywords_to_use if col in df.columns]

if not keywords_to_use_for_agg:
    print("Error: None of the specified keywords were found as columns in the DataFrame. Stopping aggregation.")
    pivot_data_grade = pd.DataFrame()
else:
    print(f"✅ 集計に使用するTF-IDF列: {keywords_to_use_for_agg}")

    # 学年別に集計
    tfidf_agg = df.groupby(analysis_target)[keywords_to_use_for_agg].mean()

    # 参加者数を計算
    if 'メールアドレス' in df.columns:
        count_agg = df.groupby(analysis_target)['メールアドレス'].size().to_frame(name='participant_count')
    else:
        print("Error: 'メールアドレス' column not found in the DataFrame. Cannot calculate participant counts.")
        count_agg = pd.DataFrame(index=df[analysis_target].unique())

    # 集計データを結合
    if not count_agg.empty:
        pivot_data_grade = tfidf_agg.join(count_agg, how='left')
    else:
        pivot_data_grade = tfidf_agg

    # TF-IDFのNaNを0に置換
    pivot_data_grade = pivot_data_grade.fillna(0)

    # 参加者2名以上の学年に絞る
    if 'participant_count' in pivot_data_grade.columns:
        pivot_data_grade = pivot_data_grade[pivot_data_grade['participant_count'] > 1].drop(columns=['participant_count']).reset_index()
    else:
        print("Warning: 'participant_count' column not available for filtering. Proceeding without filtering by participant count.")
        pivot_data_grade = pivot_data_grade.reset_index()

    print("✅ 学年別データ集計が完了しました。")

# 列名のリネーム（番号に置き換え）
tfidf_column_names = [col for col in pivot_data_grade.columns if col.startswith('tfidf_')]
num_keywords = len(tfidf_column_names)

number_labels = [str(i) for i in range(num_keywords)]
rename_map_numbers = dict(zip(tfidf_column_names, number_labels))

pivot_data_grade_final = pivot_data_grade.rename(columns=rename_map_numbers)

print("✅ データ集計と番号へのリネームが完了しました。")

# ヒートマップの可視化
plt.figure(figsize=(16, max(6, len(pivot_data_grade_final) * 0.6)))

if analysis_target in pivot_data_grade_final.columns and not pivot_data_grade_final.empty:
    heatmap_data = pivot_data_grade_final.set_index(analysis_target)

    # 横軸の順序を連番の順に固定
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

    plt.title('Grade Level Event Interest Heatmap (Keyword Index)', fontsize=16)
    plt.ylabel('Keyword Index (0, 1, 2, 3, ...)', fontsize=12)
    plt.xlabel('Grade Level', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # グラフをファイルに保存
    output_filename = 'grade/grade_event_heatmap.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ 学年別ヒートマップを '{output_filename}' に保存しました。")
    
    # GUIウィンドウを開かずにプログラムを終了
    plt.close()

else:
    print("Warning: Final data for heatmap is empty or missing the index column. Cannot plot.")

# キーワード対応表の出力
print("\n--- 💡 キーワード対応表 ---")
japanese_keywords = [kw.replace('tfidf_', '') for kw in final_keywords_to_use]
keyword_table = pd.DataFrame({
    'Index': number_labels,
    'Keyword (Japanese)': japanese_keywords
})
print(keyword_table)

# キーワード対応表をファイルに保存
keyword_table.to_csv('grade/grade_heatmap_keywords.txt', sep='\t', index=False)
print("✅ キーワード対応表を 'grade_heatmap_keywords.txt' に保存しました。")

print("\n=== 分析完了 ===")
print(f"分析対象学年: {sorted(df['学年カテゴリー'].unique())}")
print(f"使用キーワード数: {len(final_keywords_to_use)}")
print(f"参加者総数: {len(df)}")
