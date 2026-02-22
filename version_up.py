import os
import sys
import subprocess
from datetime import datetime

VERSION_FILE = "VERSION"
CHANGELOG_FILE = "CHANGELOG.md"

def get_current_version():
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, "r") as f:
            return f.read().strip()
    return "1.0.0"

def get_git_commits():
    try:
        # Lấy hash của lần cuối cùng file VERSION được commit
        last_commit_cmd = ["git", "log", "-1", "--pretty=format:%H", VERSION_FILE]
        last_hash = subprocess.run(last_commit_cmd, capture_output=True, text=True).stdout.strip()
        
        if not last_hash:
            # Nếu chưa từng commit file VERSION, lấy tất cả commit
            log_cmd = ["git", "log", "--pretty=format:- %s ([%h](https://github.com/user/repo/commit/%H))"]
        else:
            # Lấy các commit từ hash đó tới HEAD
            log_cmd = ["git", "log", f"{last_hash}..HEAD", "--pretty=format:- %s ([%h](https://github.com/user/repo/commit/%H))"]
            
        result = subprocess.run(log_cmd, capture_output=True, text=True).stdout.strip()
        # Chuyển đổi link placeholder nếu bạn muốn (ở đây để mặc định)
        return result if result else "- Minor updates and improvements."
    except Exception as e:
        return f"- No git commits found ({e})."

def update_version(how):
    current = get_current_version()
    major, minor, patch = map(int, current.split("."))
    if how == "major":
        major += 1
        minor = 0
        patch = 0
    elif how == "minor":
        minor += 1
        patch = 0
    else:
        patch += 1
    
    new_version = f"{major}.{minor}.{patch}"
    with open(VERSION_FILE, "w") as f:
        f.write(new_version)
    return current, new_version

def update_changelog(old_v, new_v, message):
    date_str = datetime.now().strftime("%Y-%m-%d")
    header = f"## [{new_v}] - {date_str}\n"
    
    # Đảm bảo format entry hợp lệ
    lines = []
    for line in message.split('\n'):
        if line.strip():
            if not line.startswith("- "):
                lines.append(f"- {line.strip()}")
            else:
                lines.append(line.strip())
    
    entry = "\n".join(lines) + "\n\n"
    
    content = ""
    if os.path.exists(CHANGELOG_FILE):
        with open(CHANGELOG_FILE, "r", encoding="utf-8") as f:
            content = f.read()
    
    if "# Changelog" not in content:
        content = "# Changelog\n\nAll notable changes to this project will be documented in this file.\n\n" + content
    
    # Chèn vào dưới phần giới thiệu đầu tiên
    title_marker = "documented in this file.\n\n"
    marker_pos = content.find(title_marker)
    if marker_pos != -1:
        insert_pos = marker_pos + len(title_marker)
        new_content = content[:insert_pos] + header + entry + content[insert_pos:]
    else:
        new_content = content + "\n" + header + entry
    
    with open(CHANGELOG_FILE, "w", encoding="utf-8") as f:
        f.write(new_content)

def auto_git_release(new_v):
    print(f"Bắt đầu quy trình Git Release cho v{new_v}...")
    try:
        subprocess.run(["git", "add", VERSION_FILE, CHANGELOG_FILE], check=True)
        commit_msg = f"chore(release): {new_v}"
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        subprocess.run(["git", "tag", "-a", f"v{new_v}", "-m", f"Release {new_v}"], check=True)
        print(f"✅ Đã tạo commit và tag v{new_v} thành công.")
    except Exception as e:
        print(f"❌ Lỗi khi thực hiện Git Release: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python version_up.py [patch|minor|major] ['Manual commit message']")
        sys.exit(1)
    
    level = sys.argv[1]
    
    if len(sys.argv) >= 3:
        msg = sys.argv[2]
    else:
        print("Đang quét Git history để tạo Changelog...")
        msg = get_git_commits()
    
    old_v, new_v = update_version(level)
    update_changelog(old_v, new_v, msg)
    
    print(f"Đã cập nhật từ {old_v} -> {new_v}.")
    
    # Tự động hóa release
    auto_git_release(new_v)
    print("\nQuy trình hoàn tất!")
