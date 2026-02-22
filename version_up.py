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
            log_cmd = ["git", "log", "--pretty=format:%s|%h|%H"]
        else:
            log_cmd = ["git", "log", f"{last_hash}..HEAD", "--pretty=format:%s|%h|%H"]
            
        result = subprocess.run(log_cmd, capture_output=True, text=True).stdout.strip()
        if not result:
            return "- Minor updates and improvements."

        # Phân loại commit theo chuẩn Conventional Commits
        categories = {
            "Features": [],
            "Bug Fixes": [],
            "Performance": [],
            "Refactor": [],
            "Documentation": [],
            "Chores": []
        }
        
        others = []
        
        for line in result.split("\n"):
            if not line.strip(): continue
            msg, short_h, long_h = line.split("|")
            link = f"([#{short_h}](https://github.com/user/repo/commit/{long_h}))"
            
            msg_lower = msg.lower()
            if msg_lower.startswith("feat"):
                categories["Features"].append(f"- {msg} {link}")
            elif msg_lower.startswith("fix"):
                categories["Bug Fixes"].append(f"- {msg} {link}")
            elif msg_lower.startswith("perf"):
                categories["Performance"].append(f"- {msg} {link}")
            elif msg_lower.startswith("refactor"):
                categories["Refactor"].append(f"- {msg} {link}")
            elif msg_lower.startswith("doc"):
                categories["Documentation"].append(f"- {msg} {link}")
            elif msg_lower.startswith("chore"):
                categories["Chores"].append(f"- {msg} {link}")
            else:
                others.append(f"- {msg} {link}")

        # Xây dựng nội dung Markdown
        output = []
        for cat, items in categories.items():
            if items:
                output.append(f"### {cat}")
                output.extend(items)
                output.append("")
        
        if others:
            if output: output.append("### Others")
            output.extend(others)
            output.append("")
            
        return "\n".join(output).strip()
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
    
    # entry đã được format sẵn ở get_git_commits
    entry = f"{message}\n\n"
    
    content = ""
    if os.path.exists(CHANGELOG_FILE):
        with open(CHANGELOG_FILE, "r", encoding="utf-8") as f:
            content = f.read()
    
    if "# Changelog" not in content:
        content = "# Changelog\n\nAll notable changes to this project will be documented in this file.\n\n" + content
    
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
