# Flask主应用
import os
import sys
from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEBUG, SECRET_KEY, PORT
from src.recommendation.recommendation_engine import MusicRecommendationEngine

# 创建Flask应用实例
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['DEBUG'] = DEBUG
app.config['SECRET_KEY'] = SECRET_KEY

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
app.logger.setLevel(logging.INFO)

# 创建推荐引擎实例（全局单例）
recommendation_engine = None


def initialize_engine():
    """
    初始化推荐引擎
    """
    global recommendation_engine
    try:
        recommendation_engine = MusicRecommendationEngine()
        app.logger.info("推荐引擎初始化成功")
    except Exception as e:
        app.logger.error(f"推荐引擎初始化失败: {e}")
        recommendation_engine = None

# 应用启动钩子已移除 - 在Flask 2.3.0+中before_first_request已弃用
# 推荐引擎在run_app()函数和应用入口处初始化


@app.route('/')
def index():
    """
    首页路由
    """
    return render_template('index.html')


@app.route('/api/recommend', methods=['POST'])
def recommend():
    """
    推荐API接口
    """
    try:
        # 获取请求数据
        data = request.get_json()
        
        # 验证输入
        genres = data.get('genres', {})
        listened_tracks = data.get('listened_tracks', [])
        top_k = data.get('top_k', 10)
        
        # 验证数据类型
        if not isinstance(genres, dict):
            return jsonify({'error': 'genres must be a dictionary'}), 400
        if not isinstance(listened_tracks, list):
            return jsonify({'error': 'listened_tracks must be a list'}), 400
        
        # 确保listened_tracks中的元素都是整数
        try:
            listened_tracks = [int(tid) for tid in listened_tracks]
        except ValueError:
            return jsonify({'error': 'track IDs must be integers'}), 400
        
        # 获取推荐结果
        recommendations = recommendation_engine.get_recommendations_with_explanations(
            genres, listened_tracks, top_k
        )
        
        # 返回推荐结果
        return jsonify({
            'recommendations': recommendations,
            'total': len(recommendations)
        })
        
    except Exception as e:
        app.logger.error(f"推荐过程中出错: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/genres', methods=['GET'])
def get_available_genres():
    """
    获取可用的音乐流派
    """
    try:
        # 直接返回标准格式的流派列表
        genres_list = [f'Genre {i}' for i in range(1, 11)]
        
        return jsonify({
            'genres': genres_list
        })
        
    except Exception as e:
        app.logger.error(f"获取流派列表出错: {e}")
        # 出错时返回标准的Genre格式流派
        fallback_genres = [f'Genre {i}' for i in range(1, 11)]
        return jsonify({
            'genres': fallback_genres
        })


@app.route('/api/tracks/search', methods=['GET'])
def search_tracks():
    """
    搜索曲目
    """
    try:
        query = request.args.get('q', '').lower()
        limit = request.args.get('limit', 10, type=int)
        
        # 搜索曲目
        results = []
        for track_id, info in recommendation_engine.track_info.items():
            if query in info['title'].lower() or query in info['artist'].lower():
                results.append({
                    'track_id': info['track_id'],
                    'title': info['title'],
                    'artist': info['artist'],
                    'release_year': info['release_year']
                })
        
        # 按流行度排序并限制数量
        results.sort(key=lambda x: recommendation_engine.track_info[f'track_{x["track_id"]}']['popularity'], reverse=True)
        results = results[:limit]
        
        return jsonify({
            'results': results,
            'total': len(results)
        })
        
    except Exception as e:
        app.logger.error(f"搜索曲目出错: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health_check():
    """
    健康检查接口
    """
    return jsonify({
        'status': 'healthy',
        'engine_initialized': recommendation_engine is not None
    })


@app.route('/favicon.ico')
def favicon():
    """
    图标路由
    """
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')


@app.errorhandler(404)
def page_not_found(error):
    """
    404错误处理
    """
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(error):
    """
    500错误处理
    """
    app.logger.error(f"服务器内部错误: {error}")
    return render_template('500.html'), 500


def run_app():
    """
    运行应用
    """
    # 初始化推荐引擎
    initialize_engine()
    
    # 运行Flask应用
    app.run(debug=DEBUG, host='0.0.0.0', port=PORT)


if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # 初始化推荐引擎
    initialize_engine()
    
    # 启动应用
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG)