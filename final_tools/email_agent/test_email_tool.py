#!/usr/bin/env python3

import sys
import os

from email_tool import send_email_tool

def test_email_sending():
    """Test the email tool with actual files"""
    
    # File paths to test
    filenames = [
        "/home/shruthi/SME-Indian-Constitution-And-Rights/agent_data/sessions/shruthi_7fa607f2c99e.json",
        "/home/shruthi/SME-Indian-Constitution-And-Rights/data/books/keps102.pdf",
        "/home/shruthi/SME-Indian-Constitution-And-Rights/data/ppt/human_rights.pptx"
    ]
    
    recipient_email = "shruthi.harmini2001@gmail.com"
    subject = "Test Email with Multiple Attachments - Constitution Documents"
    
    print("Testing email tool with the following files:")
    for i, filename in enumerate(filenames, 1):
        print(f"{i}. {filename}")
        # Check if file exists
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"   ✓ File exists ({size:,} bytes)")
        else:
            print(f"   ✗ File does not exist")
    
    print(f"\nRecipient: {recipient_email}")
    print(f"Subject: {subject}")
    print("\nSending email...")
    
    try:
        result = send_email_tool.invoke({
            "filenames": filenames,
            "recipient_email": recipient_email,
            "subject": subject
        })
        print(f"\nResult: {result}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_email_sending()
