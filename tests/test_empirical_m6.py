import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
import time
import json
import urllib.request
from tests.fixtures.generator import generate_synthetic_video
from app.media.probe import probe_video

def main():
    test_clip = 'tests/empirical_sample_480p.mp4'
    generate_synthetic_video(test_clip, width=640, height=480, fps=30.0, duration=1.0, has_audio=True)
    print('[+] Generated test clip 640x480 30fps at', test_clip)

    # 1. Test Upload via multipart form
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    with open(test_clip, 'rb') as f:
        video_bytes = f.read()

    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="empirical_sample_480p.mp4"\r\n'
        f'Content-Type: video/mp4\r\n\r\n'
    ).encode('utf-8') + video_bytes + f'\r\n--{boundary}--\r\n'.encode('utf-8')

    req = urllib.request.Request(
        'http://127.0.0.1:8009/api/upload',
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
    )
    with urllib.request.urlopen(req) as resp:
        upload_res = json.loads(resp.read().decode('utf-8'))
    print('[+] Upload response:', upload_res)
    task_id = upload_res['task_id']
    meta = upload_res['metadata']
    print(f'[+] Uploaded: {meta["width"]}x{meta["height"]} @ {meta["fps"]} fps, duration={meta["duration"]}s, has_audio={meta["has_audio"]}')

    # 2. Test Compare Frame
    req_comp = urllib.request.Request(f'http://127.0.0.1:8009/api/compare-frame/{task_id}')
    with urllib.request.urlopen(req_comp) as resp:
        comp_res = json.loads(resp.read().decode('utf-8'))
    print('[+] Compare frame ready:', comp_res['ready'])

    # 3. Process video with 1080p target, 60fps target, Quality mode
    proc_payload = json.dumps({'task_id': task_id, 'resolution': '1080p', 'fps': '60fps', 'mode': 'quality'}).encode('utf-8')
    req_proc = urllib.request.Request(
        'http://127.0.0.1:8009/api/process',
        data=proc_payload,
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req_proc) as resp:
        proc_res = json.loads(resp.read().decode('utf-8'))
    print('[+] Process started:', proc_res)

    # 4. Monitor status until completed
    for i in range(180):
        time.sleep(1)
        with urllib.request.urlopen(f'http://127.0.0.1:8009/api/status/{task_id}') as resp:
            st = json.loads(resp.read().decode('utf-8'))
        if i % 5 == 0 or st['status'] in ('completed', 'failed'):
            print(f'Progress: {st["progress"]}%, Stage: {st["stage"]}, Status: {st["status"]}')
        if st['status'] in ('completed', 'failed'):
            break

    assert st['status'] == 'completed', f'Job failed: {st.get("error")}'

    # 5. Download and verify output
    out_file = 'tests/empirical_output_1080p60.mp4'
    urllib.request.urlretrieve(f'http://127.0.0.1:8009/api/download/{task_id}', out_file)
    print('[+] Downloaded enhanced video to:', out_file)

    out_meta = probe_video(out_file)
    print(f'[+] Enhanced probe: {out_meta.width}x{out_meta.height} @ {out_meta.fps} fps, frames={out_meta.nb_frames}, duration={out_meta.duration}s, has_audio={out_meta.has_audio}')
    assert out_meta.width == 1920, f'Expected 1920 width, got {out_meta.width}'
    assert out_meta.height == 1080, f'Expected 1080 height, got {out_meta.height}'
    assert round(out_meta.fps) == 60, f'Expected 60 fps, got {out_meta.fps}'
    assert out_meta.has_audio is True, 'Expected audio track preserved'
    print('[SUCCESS] All empirical checks passed for 480p -> 1080p @ 60fps in Quality mode!')

    # Cleanup test files
    for p in [test_clip, out_file]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

if __name__ == '__main__':
    main()
