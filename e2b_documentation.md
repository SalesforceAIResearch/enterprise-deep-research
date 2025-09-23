This is my sandbox ID: xe0uinj2n1ufrgmmksbu

✅ Building sandbox template xe0uinj2n1ufrgmmksbu finished.

from e2b import Sandbox, AsyncSandbox                                                                                  
                                                                                                                        
# Create sync sandbox                                                                                                  
sandbox = Sandbox("xe0uinj2n1ufrgmmksbu")                                                                              
                                                                                                                        
# Create async sandbox                                                                                                 
sandbox = await AsyncSandbox.create("xe0uinj2n1ufrgmmksbu")   


Start your custom sandbox
Now you can use the template ID to create a sandbox with the SDK.

from e2b_code_interpreter import Sandbox

# Your template ID from the previous step
template_id = 'id-of-your-template' 
# Pass the template ID to the `Sandbox.create` method
sandbox = Sandbox(template_id) 

# The template installed cowsay, so we can use it
execution = sandbox.run_code("""
import cowsay
cowsay.say('Hello from E2B!')
""")

print(execution.stdout)



Sandbox persistence
Sandbox persistence is currently in public beta:

You'll need to install the beta version of the SDKs.
Consider some limitations.
The persistence is free for all users during the beta.
The sandbox persistence allows you to pause your sandbox and resume it later from the same state it was in when you paused it.

This includes not only state of the sandbox's filesystem but also the sandbox's memory. This means all running processes, loaded variables, data, etc.

1. Installing the beta version of the SDKs
To use the sandbox persistence, you need to install the beta version of the SDKs.


JavaScript & TypeScript

Python
pip install e2b-code-interpreter==1.2.0b1
#
# or use Core: https://github.com/e2b-dev/e2b
# pip install e2b==1.2.0b1
#
# or use Desktop: https://github.com/e2b-dev/desktop
# pip install e2b-desktop==1.2.0b1

Copy
Copied!
2. Pausing sandbox
When you pause a sandbox, both the sandbox's filesystem and memory state will be saved. This includes all the files in the sandbox's filesystem and all the running processes, loaded variables, data, etc.


JavaScript & TypeScript

Python
from e2b_code_interpreter import Sandbox
# or use Core: https://github.com/e2b-dev/e2b
# from e2b import Sandbox
#
# or use Desktop: https://github.com/e2b-dev/desktop
# from e2b_desktop import Sandbox

sbx = Sandbox()
print('Sandbox created', sbx.sandbox_id)

# Pause the sandbox
# You can save the sandbox ID in your database
# to resume the sandbox later
sandbox_id = sbx.pause() 
print('Sandbox paused', sandbox_id) 

Copy
Copied!
3. Resuming sandbox
When you resume a sandbox, it will be in the same state it was in when you paused it. This means that all the files in the sandbox's filesystem will be restored and all the running processes, loaded variables, data, etc. will be restored.


JavaScript & TypeScript

Python
from e2b import Sandbox
# or use Core: https://github.com/e2b-dev/e2b
# from e2b import Sandbox
#
# or use Desktop: https://github.com/e2b-dev/desktop
# from e2b_desktop import Sandbox

sbx = Sandbox()
print('Sandbox created', sbx.sandbox_id)

# Pause the sandbox
# You can save the sandbox ID in your database
# to resume the sandbox later
sandbox_id = sbx.pause()
print('Sandbox paused', sandbox_id)

# Resume the sandbox from the same state
same_sbx = Sandbox.resume(sandbox_id) 
print('Sandbox resumed', same_sbx.sandbox_id) 

Copy
Copied!
Sandbox's timeout
When you resume a sandbox, the sandbox's timeout is reset to the default timeout of an E2B sandbox - 5 minutes.

You can pass a custom timeout to the Sandbox.resume() method like this:


JavaScript & TypeScript

Python
from e2b_code_interpreter import Sandbox
# or use Core: https://github.com/e2b-dev/e2b
# from e2b import Sandbox
#
# or use Desktop: https://github.com/e2b-dev/desktop
# from e2b_desktop import Sandbox

sbx = Sandbox.resume(sandbox_id, timeout=60) # 60 seconds

Copy
Copied!
Network
If you have a service (for example a server) running inside your sandbox and you pause the sandbox, the service won't be accessible from the outside and all the clients will be disconnected. If you resume the sandbox, the service will be accessible again but you need to connect clients again.

Limitations while in beta
It takes about 4 seconds per 1 GB RAM to pause the sandbox
It takes about 1 second to resume the sandbox
Sandbox can be paused up to 30 days
After 30 days, the data will be deleted and you will not be able to resume the sandbox. Trying to resume sandbox that was deleted or does not exist will result in the NotFoundError error in JavaScript SDK and NotFoundException exception in Python SDK


Connect to running sandbox
If you have a running sandbox, you can connect to it using the Sandbox.connect() method and then start controlling it with our SDK.

This is useful if you want to, for example, reuse the same sandbox instance for the same user after a short period of inactivity.

1. Get the sandbox ID
To connect to a running sandbox, you first need to retrieve its ID. You can do this by calling the Sandbox.list() method.


JavaScript & TypeScript

Python
import { Sandbox } from "@e2b/code-interpreter"

// Get all running sandboxes
const runningSandboxes = await Sandbox.list() 

if (runningSandboxes.length === 0) {
  throw new Error("No running sandboxes found")
}

// Get the ID of the sandbox you want to connect to
const sandboxId = runningSandboxes[0].sandboxId

Copy
Copied!
2. Connect to the sandbox
Now that you have the sandbox ID, you can connect to the sandbox using the Sandbox.connect() method.


JavaScript & TypeScript

Python
import { Sandbox } from "@e2b/code-interpreter"

// Get all running sandboxes
const runningSandboxes = await Sandbox.list()

if (runningSandboxes.length === 0) {
  throw new Error("No running sandboxes found")
}

// Get the ID of the sandbox you want to connect to
const sandboxId = runningSandboxes[0].sandboxId

// Connect to the sandbox
const sandbox = await Sandbox.connect(sandboxId) 
// Now you can use the sandbox as usual
// ...

Read & write files
Reading files
You can read files from the sandbox filesystem using the files.read() method.


JavaScript & TypeScript

Python
from e2b_code_interpreter import Sandbox

sandbox = Sandbox()
file_content = sandbox.files.read('/path/to/file')

Copy
Copied!
Writing single files
You can write single files to the sandbox filesystem using the files.write() method.


JavaScript & TypeScript

Python
from e2b_code_interpreter import Sandbox

sandbox = Sandbox()

await sandbox.files.write('/path/to/file', 'file content')

Copy
Copied!
Writing multiple files
You can also write multiple files to the sandbox filesystem using the files.write() method.


JavaScript & TypeScript

Python
from e2b_code_interpreter import Sandbox

sandbox = Sandbox()

await sandbox.files.write([
    { "path": "/path/to/a", "data": "file content" },
    { "path": "another/path/to/b", "data": "file content" }
])

Watch sandbox directory for changes
You can watch a directory for changes using the files.watchDir() method in JavaScript and files.watch_dir() method in Python.


JavaScript & TypeScript

Python
from e2b_code_interpreter import Sandbox

sandbox = Sandbox()
dirname = '/home/user'

# Watch directory for changes
handle = sandbox.files.watch_dir(dirname) 
# Trigger file write event
sandbox.files.write(f"{dirname}/my-file", "hello")

# Retrieve the latest new events since the last `get_new_events()` call
events = handle.get_new_events() 
for event in events: 
  print(event) 
  if event.type == FilesystemEventType.Write: 
    print(f"wrote to file {event.name}") 

Copy
Copied!
Recursive Watching
You can enable recursive watching using the parameter recursive.

When rapidly creating new folders (e.g., deeply nested path of folders), events other than CREATE might not be emitted. To avoid this behavior, create the required folder structure in advance.


JavaScript & TypeScript

Python
from e2b_code_interpreter import Sandbox

sandbox = Sandbox()
dirname = '/home/user'

# Watch directory for changes
handle = sandbox.files.watch_dir(dirname, recursive=True) 
# Trigger file write event
sandbox.files.write(f"{dirname}/my-folder/my-file", "hello") 

# Retrieve the latest new events since the last `get_new_events()` call
events = handle.get_new_events()
for event in events:
  print(event)
  if event.type == FilesystemEventType.Write:
    print(f"wrote to file {event.name}")


Upload data to sandbox
You can upload data to the sandbox using the files.write() method.

Upload single file

JavaScript & TypeScript

Python
from e2b_code_interpreter import Sandbox

sandbox = Sandbox()

# Read file from local filesystem
with open("path/to/local/file", "rb") as file:
  # Upload file to sandbox
  sandbox.files.write("/path/in/sandbox", file)

Copy
Copied!
Upload directory / multiple files

JavaScript & TypeScript

Python
import os
from e2b_code_interpreter import Sandbox

sandbox = Sandbox()

def read_directory_files(directory_path):
    files = []
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # Skip if it's a directory
        if os.path.isfile(file_path):
            # Read file contents in binary mode
            with open(file_path, "rb") as file:
                files.append({
                    'path': file_path,
                    'data': file.read()
                })
    
    return files

files = read_directory_files("/local/dir")
print(files)
# [
#  {"'path": "/local/dir/file1.txt", "data": "File 1 contents..." },
#   { "path": "/local/dir/file2.txt", "data": "File 2 contents..." },
#   ...
# ]

sandbox.files.write(files)


Download data from sandbox
You can download data from the sandbox using the files.read() method.


JavaScript & TypeScript

Python
from e2b_code_interpreter import Sandbox

sandbox = Sandbox()

# Read file from sandbox
content = sandbox.files.read('/path/in/sandbox')
# Write file to local filesystem
with open('/local/path', 'w') as file:
  file.write(content)