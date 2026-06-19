#!/usr/bin/env python3
"""
Agentic IDE with AI Assistant
Features:
- Code Editor with syntax highlighting
- File Explorer
- AI Assistant using Z.AI GLM-4.7 Flash
- Settings management for API key
- Rate limit protection
- File editing and viewing capabilities
"""

import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any

# Try to import optional dependencies
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from pygments import highlight
    from pygments.lexers import get_lexer_for_filename, PythonLexer
    from pygments.formatters import TerminalFormatter
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False


class RateLimiter:
    """Rate limiter to prevent API rate limit issues"""
    
    def __init__(self, max_requests: int = 10, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window  # seconds
        self.requests: List[datetime] = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if we're approaching the rate limit"""
        with self.lock:
            now = datetime.now()
            # Remove old requests outside the time window
            self.requests = [req for req in self.requests 
                           if now - req < timedelta(seconds=self.time_window)]
            
            if len(self.requests) >= self.max_requests:
                # Calculate wait time
                oldest_request = self.requests[0]
                wait_time = self.time_window - (now - oldest_request).total_seconds()
                if wait_time > 0:
                    print(f"\n⏳ Rate limit approaching. Waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    # Clean up again after waiting
                    now = datetime.now()
                    self.requests = [req for req in self.requests 
                                   if now - req < timedelta(seconds=self.time_window)]
            
            self.requests.append(datetime.now())
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        with self.lock:
            now = datetime.now()
            self.requests = [req for req in self.requests 
                           if now - req < timedelta(seconds=self.time_window)]
            return {
                'requests_made': len(self.requests),
                'max_requests': self.max_requests,
                'time_window': self.time_window,
                'remaining': self.max_requests - len(self.requests)
            }


class GLM47FlashClient:
    """Client for Z.AI GLM-4.7 Flash API"""
    
    # Official free tier endpoint for GLM-4.7 Flash
    API_BASE_URL = "https://api.z.ai/api/paas/v4/chat/completions"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rate_limiter = RateLimiter(max_requests=15, time_window=60)
        self.model = "glm-4-flash"
    
    def chat(self, messages: List[Dict[str, str]], 
             system_prompt: Optional[str] = None,
             temperature: float = 0.7,
             max_tokens: int = 2048) -> Optional[str]:
        """Send a chat request to GLM-4.7 Flash"""
        
        if not REQUESTS_AVAILABLE:
            print("Error: 'requests' library not installed. Run: pip install requests")
            return None
        
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare messages
        all_messages = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)
        
        payload = {
            "model": self.model,
            "messages": all_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            response = requests.post(
                self.API_BASE_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('choices', [{}])[0].get('message', {}).get('content', '')
            elif response.status_code == 429:
                print("\n⚠️  Rate limit exceeded! Waiting 30 seconds before retry...")
                time.sleep(30)
                return self.chat(messages, system_prompt, temperature, max_tokens)
            else:
                print(f"\n❌ API Error: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"\n❌ Request failed: {str(e)}")
            return None
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        return self.rate_limiter.get_status()


class SettingsManager:
    """Manages IDE settings including API keys"""
    
    SETTINGS_FILE = ".agentic_ide_settings.json"
    
    def __init__(self):
        self.settings = self.load_settings()
    
    def load_settings(self) -> Dict[str, Any]:
        """Load settings from file"""
        if os.path.exists(self.SETTINGS_FILE):
            try:
                with open(self.SETTINGS_FILE, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {
            'api_key': '',
            'working_directory': os.getcwd(),
            'theme': 'dark',
            'auto_save': True
        }
    
    def save_settings(self):
        """Save settings to file"""
        with open(self.SETTINGS_FILE, 'w') as f:
            json.dump(self.settings, f, indent=2)
    
    def set_api_key(self, api_key: str):
        """Set the API key"""
        self.settings['api_key'] = api_key
        self.save_settings()
    
    def get_api_key(self) -> str:
        """Get the API key"""
        return self.settings.get('api_key', '')
    
    def set_working_directory(self, path: str):
        """Set the working directory"""
        self.settings['working_directory'] = path
        self.save_settings()
    
    def get_working_directory(self) -> str:
        """Get the working directory"""
        return self.settings.get('working_directory', os.getcwd())


class FileManager:
    """Handles file operations"""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
    
    def list_files(self, path: Optional[str] = None, 
                   show_hidden: bool = False) -> List[Dict[str, Any]]:
        """List files and directories"""
        current_path = Path(path) if path else self.root_path
        
        if not current_path.exists():
            return []
        
        items = []
        try:
            for item in current_path.iterdir():
                if not show_hidden and item.name.startswith('.'):
                    continue
                
                item_info = {
                    'name': item.name,
                    'path': str(item),
                    'is_directory': item.is_dir(),
                    'size': item.stat().st_size if item.is_file() else 0,
                    'modified': datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                }
                items.append(item_info)
        except PermissionError:
            pass
        
        # Sort: directories first, then files
        items.sort(key=lambda x: (not x['is_directory'], x['name'].lower()))
        return items
    
    def read_file(self, file_path: str) -> Optional[str]:
        """Read file content"""
        try:
            path = Path(file_path)
            if not path.exists():
                return None
            
            # Check file size (limit to 1MB for safety)
            if path.stat().st_size > 1024 * 1024:
                print("⚠️  File too large to display (>1MB)")
                return None
            
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
    
    def write_file(self, file_path: str, content: str) -> bool:
        """Write content to file"""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"Error writing file: {e}")
            return False
    
    def create_directory(self, dir_path: str) -> bool:
        """Create a directory"""
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"Error creating directory: {e}")
            return False
    
    def delete_file(self, file_path: str) -> bool:
        """Delete a file"""
        try:
            Path(file_path).unlink()
            return True
        except Exception as e:
            print(f"Error deleting file: {e}")
            return False


class CodeEditor:
    """Simple code editor with syntax highlighting support"""
    
    def __init__(self):
        self.current_file: Optional[str] = None
        self.content: str = ""
        self.modified: bool = False
    
    def open_file(self, file_path: str, file_manager: FileManager) -> bool:
        """Open a file for editing"""
        content = file_manager.read_file(file_path)
        if content is None:
            return False
        
        self.current_file = file_path
        self.content = content
        self.modified = False
        return True
    
    def set_content(self, content: str):
        """Set editor content"""
        self.content = content
        self.modified = True
    
    def save(self, file_manager: FileManager) -> bool:
        """Save current file"""
        if not self.current_file:
            return False
        
        success = file_manager.write_file(self.current_file, self.content)
        if success:
            self.modified = False
        return success
    
    def display_content(self, use_syntax_highlighting: bool = True):
        """Display file content with optional syntax highlighting"""
        if not self.content:
            print("(empty file)")
            return
        
        if use_syntax_highlighting and PYGMENTS_AVAILABLE and self.current_file:
            try:
                lexer = get_lexer_for_filename(self.current_file)
                highlighted = highlight(self.content, lexer, TerminalFormatter())
                print(highlighted)
                return
            except:
                pass
        
        # Fallback: plain text with line numbers
        lines = self.content.split('\n')
        for i, line in enumerate(lines, 1):
            print(f"{i:4d} | {line}")


class AIAgent:
    """AI Agent that can interact with files and provide assistance"""
    
    SYSTEM_PROMPT = """You are an intelligent coding assistant integrated into an IDE. 
You can help users with:
- Writing and editing code
- Explaining code functionality
- Debugging issues
- Suggesting improvements
- Creating new files
- Reviewing code

When asked to edit or create files, provide clear instructions or the complete code.
Always be concise and helpful."""
    
    def __init__(self, client: GLM47FlashClient, file_manager: FileManager, editor: CodeEditor):
        self.client = client
        self.file_manager = file_manager
        self.editor = editor
        self.conversation_history: List[Dict[str, str]] = []
    
    def chat(self, user_message: str, context: Optional[str] = None) -> Optional[str]:
        """Send a message to the AI and get a response"""
        
        # Build context-aware message
        full_message = user_message
        if context:
            full_message = f"Context:\n{context}\n\nQuestion: {user_message}"
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": full_message})
        
        # Keep only last 10 messages to avoid token limits
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        response = self.client.chat(
            messages=self.conversation_history,
            system_prompt=self.SYSTEM_PROMPT
        )
        
        if response:
            self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def analyze_file(self, file_path: str) -> Optional[str]:
        """Ask AI to analyze a file"""
        content = self.file_manager.read_file(file_path)
        if not content:
            return None
        
        context = f"File: {file_path}\n\nContent:\n{content}"
        return self.chat("Please analyze this file and explain what it does.", context)
    
    def edit_file_with_ai(self, file_path: str, instruction: str) -> bool:
        """Use AI to help edit a file based on instructions"""
        content = self.file_manager.read_file(file_path)
        if not content:
            print(f"Cannot read file: {file_path}")
            return False
        
        prompt = f"""I need to edit this file according to these instructions: {instruction}

Current file content:
{content}

Please provide the COMPLETE updated file content. Only output the code, no explanations."""
        
        response = self.chat(prompt)
        
        if response:
            # Extract code from response (remove markdown code blocks if present)
            code = response
            if "```" in response:
                # Extract content between code blocks
                parts = response.split("```")
                if len(parts) >= 2:
                    code = parts[1]
                    # Remove language identifier if present
                    if '\n' in code:
                        code = code.split('\n', 1)[1]
            
            # Save the edited content
            if self.file_manager.write_file(file_path, code):
                print(f"✓ File updated successfully: {file_path}")
                return True
        
        return False
    
    def create_file_with_ai(self, file_path: str, description: str) -> bool:
        """Use AI to create a new file based on description"""
        prompt = f"""Create a file at path: {file_path}

Description of what should be in the file:
{description}

Please provide the COMPLETE file content. Only output the code/content, no explanations."""
        
        response = self.chat(prompt)
        
        if response:
            # Extract code from response
            code = response
            if "```" in response:
                parts = response.split("```")
                if len(parts) >= 2:
                    code = parts[1]
                    if '\n' in code:
                        code = code.split('\n', 1)[1]
            
            if self.file_manager.write_file(file_path, code):
                print(f"✓ File created successfully: {file_path}")
                return True
        
        return False
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


class FileExplorerUI:
    """Text-based file explorer UI"""
    
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
        self.current_path: Path = Path(file_manager.root_path)
    
    def display(self):
        """Display current directory contents"""
        print("\n" + "="*60)
        print(f"📁 {self.current_path}")
        print("="*60)
        
        items = self.file_manager.list_files(str(self.current_path))
        
        if not items:
            print("  (empty directory)")
            return
        
        for item in items:
            icon = "📁" if item['is_directory'] else "📄"
            size_str = "" if item['is_directory'] else f" ({item['size']:,} bytes)"
            print(f"  {icon} {item['name']}{size_str}")
    
    def navigate(self, path: str) -> bool:
        """Navigate to a directory"""
        if path == "..":
            parent = self.current_path.parent
            if parent != self.current_path:
                self.current_path = parent
                return True
            return False
        
        new_path = self.current_path / path
        if new_path.exists() and new_path.is_dir():
            self.current_path = new_path
            return True
        
        print(f"Directory not found: {path}")
        return False
    
    def get_current_path(self) -> str:
        """Get current path as string"""
        return str(self.current_path)


class AgenticIDE:
    """Main IDE application"""
    
    def __init__(self):
        self.settings = SettingsManager()
        self.file_manager = FileManager(self.settings.get_working_directory())
        self.editor = CodeEditor()
        self.explorer = FileExplorerUI(self.file_manager)
        self.ai_client: Optional[GLM47FlashClient] = None
        self.ai_agent: Optional[AIAgent] = None
        self.running = True
    
    def initialize_ai(self):
        """Initialize AI client with API key"""
        api_key = self.settings.get_api_key()
        
        if not api_key:
            print("\n⚠️  No API key configured. Please set it in settings.")
            return False
        
        self.ai_client = GLM47FlashClient(api_key)
        self.ai_agent = AIAgent(self.ai_client, self.file_manager, self.editor)
        print("✓ AI Assistant initialized successfully!")
        return True
    
    def show_menu(self):
        """Display main menu"""
        print("\n" + "="*60)
        print("🚀 AGENTIC IDE - Powered by GLM-4.7 Flash")
        print("="*60)
        print("📂 File Operations:")
        print("  1. Open File")
        print("  2. Save File")
        print("  3. Create New File")
        print("  4. Delete File")
        print("\n🗂️  File Explorer:")
        print("  5. Browse Directory")
        print("  6. Change Directory")
        print("\n🤖 AI Assistant:")
        print("  7. Chat with AI")
        print("  8. Analyze File with AI")
        print("  9. Edit File with AI")
        print("  10. Create File with AI")
        print("  11. View Rate Limit Status")
        print("\n⚙️  Settings:")
        print("  12. Set API Key")
        print("  13. View Settings")
        print("\n💻 Editor:")
        print("  14. View Current File")
        print("  15. Edit Current File")
        print("\n❌ Exit: 0")
        print("="*60)
    
    def cmd_open_file(self):
        """Open a file"""
        file_path = input("Enter file path: ").strip()
        if not file_path:
            return
        
        if os.path.isabs(file_path):
            path = file_path
        else:
            path = os.path.join(self.explorer.get_current_path(), file_path)
        
        if self.editor.open_file(path, self.file_manager):
            print(f"✓ Opened: {path}")
            self.editor.display_content()
        else:
            print(f"✗ Failed to open: {path}")
    
    def cmd_save_file(self):
        """Save current file"""
        if not self.editor.current_file:
            print("No file open")
            return
        
        if self.editor.save(self.file_manager):
            print(f"✓ Saved: {self.editor.current_file}")
        else:
            print("✗ Failed to save")
    
    def cmd_create_file(self):
        """Create a new file"""
        file_path = input("Enter file path: ").strip()
        if not file_path:
            return
        
        if os.path.isabs(file_path):
            path = file_path
        else:
            path = os.path.join(self.explorer.get_current_path(), file_path)
        
        print("Enter content (type 'EOF' on a new line to finish):")
        lines = []
        while True:
            line = input()
            if line == 'EOF':
                break
            lines.append(line)
        
        content = '\n'.join(lines)
        if self.file_manager.write_file(path, content):
            print(f"✓ Created: {path}")
        else:
            print(f"✗ Failed to create: {path}")
    
    def cmd_delete_file(self):
        """Delete a file"""
        file_path = input("Enter file path to delete: ").strip()
        if not file_path:
            return
        
        confirm = input(f"Are you sure you want to delete '{file_path}'? (y/N): ")
        if confirm.lower() != 'y':
            return
        
        if os.path.isabs(file_path):
            path = file_path
        else:
            path = os.path.join(self.explorer.get_current_path(), file_path)
        
        if self.file_manager.delete_file(path):
            print(f"✓ Deleted: {path}")
        else:
            print(f"✗ Failed to delete: {path}")
    
    def cmd_browse_directory(self):
        """Browse directory"""
        self.explorer.display()
    
    def cmd_change_directory(self):
        """Change current directory"""
        path = input("Enter directory path (or '..' for parent): ").strip()
        if path:
            self.explorer.navigate(path)
            self.explorer.display()
    
    def cmd_chat_with_ai(self):
        """Chat with AI assistant"""
        if not self.ai_agent:
            if not self.initialize_ai():
                return
        
        print("\n🤖 AI Chat (type 'quit' to exit)")
        print("-" * 40)
        
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            # Add context if a file is open
            context = None
            if self.editor.current_file and self.editor.content:
                context = f"Currently editing: {self.editor.current_file}\n\n{self.editor.content[:1000]}"
            
            response = self.ai_agent.chat(user_input, context)
            
            if response:
                print(f"\nAI: {response}")
            else:
                print("\n⚠️  Failed to get response from AI")
    
    def cmd_analyze_file(self):
        """Analyze a file with AI"""
        if not self.ai_agent:
            if not self.initialize_ai():
                return
        
        file_path = input("Enter file path to analyze: ").strip()
        if not file_path:
            return
        
        if os.path.isabs(file_path):
            path = file_path
        else:
            path = os.path.join(self.explorer.get_current_path(), file_path)
        
        print("\n🔍 Analyzing file...")
        response = self.ai_agent.analyze_file(path)
        
        if response:
            print(f"\n{response}")
        else:
            print("✗ Failed to analyze file")
    
    def cmd_edit_file_with_ai(self):
        """Edit a file using AI"""
        if not self.ai_agent:
            if not self.initialize_ai():
                return
        
        file_path = input("Enter file path to edit: ").strip()
        if not file_path:
            return
        
        if os.path.isabs(file_path):
            path = file_path
        else:
            path = os.path.join(self.explorer.get_current_path(), file_path)
        
        instruction = input("Describe the changes you want: ").strip()
        if not instruction:
            return
        
        print("\n✏️  Editing file with AI...")
        if self.ai_agent.edit_file_with_ai(path, instruction):
            # Open the edited file
            self.editor.open_file(path, self.file_manager)
            self.editor.display_content()
        else:
            print("✗ Failed to edit file")
    
    def cmd_create_file_with_ai(self):
        """Create a file using AI"""
        if not self.ai_agent:
            if not self.initialize_ai():
                return
        
        file_path = input("Enter file path to create: ").strip()
        if not file_path:
            return
        
        if os.path.isabs(file_path):
            path = file_path
        else:
            path = os.path.join(self.explorer.get_current_path(), file_path)
        
        description = input("Describe what should be in the file: ").strip()
        if not description:
            return
        
        print("\n✨ Creating file with AI...")
        if self.ai_agent.create_file_with_ai(path, description):
            # Open the created file
            self.editor.open_file(path, self.file_manager)
            self.editor.display_content()
        else:
            print("✗ Failed to create file")
    
    def cmd_view_rate_limit(self):
        """View rate limit status"""
        if not self.ai_client:
            print("AI not initialized")
            return
        
        status = self.ai_client.get_rate_limit_status()
        print("\n📊 Rate Limit Status:")
        print(f"  Requests made: {status['requests_made']}/{status['max_requests']}")
        print(f"  Time window: {status['time_window']} seconds")
        print(f"  Remaining: {status['remaining']}")
    
    def cmd_set_api_key(self):
        """Set API key"""
        api_key = input("Enter your Z.AI API key: ").strip()
        if api_key:
            self.settings.set_api_key(api_key)
            print("✓ API key saved!")
            self.initialize_ai()
        else:
            print("✗ API key cannot be empty")
    
    def cmd_view_settings(self):
        """View current settings"""
        print("\n⚙️  Current Settings:")
        print(f"  Working Directory: {self.settings.get_working_directory()}")
        print(f"  API Key Configured: {'Yes' if self.settings.get_api_key() else 'No'}")
        print(f"  Theme: {self.settings.settings.get('theme', 'dark')}")
        print(f"  Auto-save: {self.settings.settings.get('auto_save', True)}")
    
    def cmd_view_current_file(self):
        """View current file content"""
        if not self.editor.current_file:
            print("No file open")
            return
        
        print(f"\n📄 {self.editor.current_file}")
        print("-" * 60)
        self.editor.display_content()
    
    def cmd_edit_current_file(self):
        """Edit current file manually"""
        if not self.editor.current_file:
            print("No file open")
            return
        
        print("\n✏️  Edit mode (type 'EOF' on a new line to save):")
        print("-" * 60)
        lines = []
        while True:
            line = input()
            if line == 'EOF':
                break
            lines.append(line)
        
        self.editor.set_content('\n'.join(lines))
        print("\nChanges applied. Don't forget to save (option 2)!")
    
    def run(self):
        """Main application loop"""
        print("\n" + "="*60)
        print("🚀 Welcome to Agentic IDE!")
        print("="*60)
        
        # Try to initialize AI if API key exists
        if self.settings.get_api_key():
            self.initialize_ai()
        
        while self.running:
            self.show_menu()
            
            try:
                choice = input("\nEnter your choice (0-15): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye! 👋")
                break
            
            if choice == '0':
                print("\nGoodbye! 👋")
                break
            elif choice == '1':
                self.cmd_open_file()
            elif choice == '2':
                self.cmd_save_file()
            elif choice == '3':
                self.cmd_create_file()
            elif choice == '4':
                self.cmd_delete_file()
            elif choice == '5':
                self.cmd_browse_directory()
            elif choice == '6':
                self.cmd_change_directory()
            elif choice == '7':
                self.cmd_chat_with_ai()
            elif choice == '8':
                self.cmd_analyze_file()
            elif choice == '9':
                self.cmd_edit_file_with_ai()
            elif choice == '10':
                self.cmd_create_file_with_ai()
            elif choice == '11':
                self.cmd_view_rate_limit()
            elif choice == '12':
                self.cmd_set_api_key()
            elif choice == '13':
                self.cmd_view_settings()
            elif choice == '14':
                self.cmd_view_current_file()
            elif choice == '15':
                self.cmd_edit_current_file()
            else:
                print("Invalid choice. Please try again.")
            
            time.sleep(0.5)


def main():
    """Entry point"""
    print("\n" + "="*60)
    print("Starting Agentic IDE...")
    print("="*60)
    
    # Check for required dependencies
    if not REQUESTS_AVAILABLE:
        print("\n⚠️  Installing required dependency: requests")
        os.system("pip install requests")
        import requests
        global REQUESTS_AVAILABLE
        REQUESTS_AVAILABLE = True
    
    # Optional: Install pygments for syntax highlighting
    if not PYGMENTS_AVAILABLE:
        print("\n💡 Tip: Install 'pygments' for syntax highlighting:")
        print("   pip install pygments")
    
    # Start the IDE
    ide = AgenticIDE()
    ide.run()


if __name__ == "__main__":
    main()
