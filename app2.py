# app2.py
import sys
import os
from flask import Flask, render_template, send_from_directory, jsonify, request, session, redirect, url_for
import cv2
import numpy as np
import base64
import json
import uuid
import time
# # --- 기본 경로 설정 (기존 프로젝트와 공유) ---
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
# UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
# # ✨ 1. pseudo mask가 있는 MASK_FOLDER 경로 추가
# MASK_FOLDER = os.path.join(BASE_DIR, 'masks') 
# ANNOTATED_FOLDER = os.path.join(BASE_DIR, 'annotated_masks2')
# CAPTION_FOLDER = os.path.join(BASE_DIR, 'caption2')

# # --- 폴더 생성 ---
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(ANNOTATED_FOLDER, exist_ok=True)
# os.makedirs(CAPTION_FOLDER, exist_ok=True)

# app = Flask(__name__, template_folder=TEMPLATE_DIR)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MASK_FOLDER = os.path.join(BASE_DIR, 'masks') 

# ✨ 1. 이 부분만 수정하면 됩니다.
# 영구 저장소의 기본 경로를 Render에서 수정한 Mount Path와 동일하게 맞춥니다.
DATA_DIR = '/var/data/annotations' 
ANNOTATED_FOLDER = os.path.join(DATA_DIR, 'annotated_masks2')
CAPTION_FOLDER = os.path.join(DATA_DIR, 'captions2')

# --- 폴더 생성 (이하 코드는 변경 없음) ---
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)
os.makedirs(CAPTION_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.secret_key = 'manual-annotation-secret-key'

# --- 사용자 ID 할당 ---
@app.before_request
def assign_user_id():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        print(f"[SESSION] New user for manual annotation. ID: {session['user_id']}")

# --- 유틸리티 함수 ---
def get_all_images():
    """uploads 폴더의 모든 이미지 목록 반환"""
    return sorted([f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

def get_image_stem(filename):
    """확장자 제거한 파일명"""
    return os.path.splitext(filename)[0]

def get_highlighted_image_base64(img_path, mask_path):
    """마스크 영역을 이미지 위에 반투명하게 오버레이"""
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # 마스크가 있는 경우에만 오버레이
    if mask is not None:
        overlay = img.copy()
        # 파란색 오버레이 생성
        blue_overlay = np.full(img.shape, (255, 150, 0), dtype=np.uint8)
        # 마스크 영역에만 적용
        overlay = np.where(mask[..., None] > 127, blue_overlay, img)
        blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    else:
        blended = img

    _, buf = cv2.imencode(".png", blended)
    return base64.b64encode(buf.tobytes()).decode("utf-8")

# ===== 라우트 (Routes) =====

@app.route('/')
def index():
    """메인 페이지 - 사용자의 다음 작업을 결정"""
    user_id = session['user_id']
    print(f"\n[INDEX] Checking progress for User: {user_id}")
    
    images = get_all_images()
    if not images:
        return "No images found in uploads folder", 404
        
    for img_file in images:
        stem = get_image_stem(img_file)
        user_mask_path = os.path.join(ANNOTATED_FOLDER, user_id, f"{stem}.png")
        user_caption_path = os.path.join(CAPTION_FOLDER, user_id, f"{stem}.json")

        if not os.path.exists(user_mask_path):
            print(f" -> User '{user_id}' needs to MASK annotate '{img_file}'")
            return redirect(url_for('mask_annotate', image_name=img_file))
        
        if not os.path.exists(user_caption_path):
            print(f" -> User '{user_id}' needs to TEXT annotate '{img_file}'")
            return redirect(url_for('text_annotate', image_name=img_file))

    print(f" -> User '{user_id}' has completed all annotations!")
    return "<h1>All annotations are complete!</h1>"

@app.route('/mask_annotate/<image_name>')
def mask_annotate(image_name):
    """수동 마스크 어노테이션 페이지 렌더링 및 가이드 점, 이미지 번호 전달"""
    stem = get_image_stem(image_name)
    pseudo_mask_path = os.path.join(MASK_FOLDER, stem, 'pseudo_mask.png')
    
    center_x, center_y = -1, -1

    if os.path.exists(pseudo_mask_path):
        mask = cv2.imread(pseudo_mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            M = cv2.moments(mask)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])

    # ✨ 1. 전체 이미지 목록을 가져와서 현재 인덱스와 전체 개수를 계산
    all_images = get_all_images()
    try:
        # 사용자가 보기 편하도록 1부터 시작하는 인덱스로 변환
        current_index = all_images.index(image_name) + 1
    except ValueError:
        current_index = 0 # 이미지를 찾지 못했을 경우의 예외 처리
    total_images = len(all_images)

    # ✨ 2. 계산된 이미지 번호와 전체 개수를 템플릿으로 전달
    return render_template(
        'mask_annotate2.html', 
        image_name=image_name,
        center_x=center_x,
        center_y=center_y,
        current_index=current_index,
        total_images=total_images
    )

@app.route('/api/save_mask/<image_stem>', methods=['POST'])
def save_mask(image_stem):
    """폴리곤 파트들과 '마스크 소요 시간', '마스터 시작 시간'을 받아 세션에 기록"""
    user_id = session['user_id']
    data = request.get_json()
    parts = data.get('parts')
    img_dims = data.get('dimensions')
    # ✨ Receive both time values
    mask_duration = data.get('mask_duration')
    master_start_time = data.get('master_start_time')

    if not parts or not img_dims:
        return jsonify({"error": "Missing parts or dimensions"}), 400

    mask = np.zeros((img_dims['h'], img_dims['w']), dtype=np.uint8)
    for part in parts:
        if len(part) >= 3:
            polygon = np.array(part, dtype=np.int32)
            cv2.fillPoly(mask, [polygon], 255)

    user_mask_dir = os.path.join(ANNOTATED_FOLDER, user_id)
    os.makedirs(user_mask_dir, exist_ok=True)
    save_path = os.path.join(user_mask_dir, f"{image_stem}.png")
    cv2.imwrite(save_path, mask)
    
    # ✨ Store both time values in the session
    session[f'{user_id}_{image_stem}_mask_duration'] = mask_duration
    session[f'{user_id}_{image_stem}_master_start_time'] = master_start_time
    print(f"[SAVE_MASK] Saved mask for user '{user_id}'. Duration: {mask_duration} ms.")
    
    original_image = next((f for f in get_all_images() if get_image_stem(f) == image_stem), None)
    redirect_url = url_for('text_annotate', image_name=original_image)
    
    return jsonify({"success": True, "redirect_url": redirect_url})

@app.route('/text_annotate/<image_name>', methods=['GET', 'POST'])
def text_annotate(image_name):
    user_id = session['user_id']
    img_path = os.path.join(UPLOAD_FOLDER, image_name)
    stem = get_image_stem(image_name)
    mask_path = os.path.join(ANNOTATED_FOLDER, user_id, f"{stem}.png")

    if not os.path.exists(mask_path):
        return redirect(url_for('mask_annotate', image_name=image_name))

    if request.method == 'POST':
        # ✨ 1. Get the final end time
        end_time_ms = int(time.time() * 1000)
        
        # ✨ 2. Retrieve both master start time and mask duration from session
        mask_duration_ms = session.pop(f'{user_id}_{stem}_mask_duration', 0)
        master_start_time_ms = session.pop(f'{user_id}_{stem}_master_start_time', 0)

        # ✨ 3. Calculate total and text duration
        total_duration_ms = end_time_ms - master_start_time_ms if master_start_time_ms > 0 else 0
        text_duration_ms = total_duration_ms - mask_duration_ms if total_duration_ms > mask_duration_ms else 0

        caption = request.form.get('caption', '').strip()
        
        user_caption_dir = os.path.join(CAPTION_FOLDER, user_id)
        os.makedirs(user_caption_dir, exist_ok=True)
        caption_path = os.path.join(user_caption_dir, f"{stem}.json")

        final_data = {
            "image": image_name,
            "caption": caption,
            "annotation_times_ms": {
                "mask": mask_duration_ms,
                "text": text_duration_ms
            }
        }

        with open(caption_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        
        print(f"[SAVE_CAPTION] Saved caption for user '{user_id}'. Total time: {total_duration_ms} ms.")
        return redirect(url_for('index'))

    image_base64 = get_highlighted_image_base64(img_path, mask_path)
    return render_template('text_annotate2.html', image_name=image_name, image_base64=image_base64)

# --- 정적 파일 라우팅 ---
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(port=3001, debug=True, use_reloader=False)