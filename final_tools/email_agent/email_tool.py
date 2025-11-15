from langchain_core.tools import tool
from typing import List, Union
import os
from .email_impl import send_email
from langchain_core.globals import set_verbose, set_debug

# To enable verbose output
set_verbose(True)
set_debug(True)

@tool
def send_email_tool(
    filenames: List[str], 
    recipient_email: str, 
    subject: str = "Report"
) -> str:
    """
    The capability of the tool is to send an email with file attachment(s) to a specified recipient.
    
    Args:
        filenames: Path to a single file or list of file paths to attach
        recipient_email: Email address of the recipient
        subject: Email subject line (defaults to "Report")
        
    Returns:
        str: Success or error message
    """
    try:
        print(f"Email tool: {filenames}")
        if isinstance(filenames, str):
            file_list = [filenames]
        else:
            file_list = filenames
        
        # Check if all files exist
        missing_files = []
        existing_files = []
        
        for filename in file_list:
            if not os.path.isabs(filename):
                filename = os.path.abspath(filename)
                
            if not os.path.exists(filename):
                missing_files.append(filename)
            else:
                existing_files.append(filename)
        
        if missing_files:
            return f"Error: The following files do not exist: {', '.join(missing_files)}"
        
        if not existing_files:
            return "Error: No valid files to attach."
        
        # Call the email implementation
        send_email(existing_files, recipient_email, subject)
        
        file_count = len(existing_files)
        attachment_names = [os.path.basename(f) for f in existing_files]
        
        if file_count == 1:
            return f"Email with subject '{subject}' successfully sent to {recipient_email} with attachment: {attachment_names[0]}"
        else:
            return f"Email with subject '{subject}' successfully sent to {recipient_email} with {file_count} attachments: {', '.join(attachment_names)}"
            
    except Exception as e:
        return f"Error sending email: {str(e)}"
