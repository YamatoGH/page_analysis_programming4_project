import requests
from bs4 import BeautifulSoup
from django.shortcuts import render
from janome.tokenizer import Tokenizer
from collections import Counter
from gensim.models import Word2Vec
import io
import base64
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib

def analyze_text(text):
    tokenizer = Tokenizer()
    tokens = [token.surface for token in tokenizer.tokenize(text)]
    freq = Counter(tokens)
    return freq.most_common(10)

def train_word2vec(text):
    tokenizer = Tokenizer()
    corpus = []
    
    # 文は改行ごと
    for line in text.splitlines():
        line = line.strip()
        if line:  # 空行は無視
            # 各行をトークン化（表層形を利用）
            tokens = [token.surface for token in tokenizer.tokenize(line)]
            corpus.append(tokens)

            # 頻度も取得しとく
            freq = Counter(tokens)
    
    # corpusが存在する場合、Word2Vecモデルを学習
    if corpus:
        model = Word2Vec(
            sentences=corpus,
            vector_size=100,  
            window=5,         
            min_count=1,      
            workers=-1
        )
        return model
    else:
        return None

def generate_tsne_plot(model):
    # プロットのラベルに日本語が対応するためにフォントを設定
    matplotlib.rcParams['font.family'] = ['IPAexGothic', 'Yu Gothic', 'Meiryo', 'sans-serif']
 
    # モデル内の単語リスト（出現頻度順にソート済み）
    words = list(model.wv.index_to_key)
    vectors = model.wv[words]
   
    # t-SNEで2次元に削減
    tsne = TSNE(n_components=2, random_state=0)
    vectors_2d = tsne.fit_transform(vectors)
    
    # プロット作成
    plt.figure(figsize=(10, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])

    # 各点に単語ラベルをつける
    for i, word in enumerate(words):
        plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=9)
    
    # 画像をメモリ上に保存
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_png = buf.getvalue()
    plot_image = base64.b64encode(image_png).decode('utf-8') #base64を使う
    buf.close()
    plt.close()
    
    return plot_image

def index(request):
    result_text = None
    error_message = None
    plot_image = None

    if request.method == 'POST':
        url = request.POST.get('url')

        if url:
            try:
                response = requests.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                result_text = soup.get_text(separator='\n', strip=True)
            except requests.exceptions.RequestException as e:
                error_message = f'ページの取得に失敗しました: {e}'
        else:
            error_message = 'URLを入力してください。'

        if result_text:
            model = train_word2vec(result_text)
            if model:
                try:
                    plot_image = generate_tsne_plot(model)
                except Exception as e:
                    error_message = f't-SNEプロット作成でエラーが発生しました: {e}'
            else:
                error_message = 'Word2Vecモデルの学習に失敗しました'
    
    return render(request, 'page_analysis_app/index.html', {
        'result_text': result_text,
        'error_message': error_message,
        'plot_image': plot_image,
    })