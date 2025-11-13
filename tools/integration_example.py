"""
Integration Example: DocumentGenerationTool + EmailAutomationTool

This example demonstrates a complete workflow:
1. Generate a report using DocumentGenerationTool
2. Send the generated report via email using EmailAutomationTool
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import (
    create_document_generation_tool,
    create_email_automation_tool
)


def generate_and_send_report():
    """
    Complete workflow: Generate a report and send it via email.
    """
    print("=" * 60)
    print("Complete Workflow: Generate Report & Send Email")
    print("=" * 60)
    
    # Step 1: Generate the report
    print("\n1. Generating report document...")
    doc_tool = create_document_generation_tool()
    
    report_content = """
    # Monthly Performance Report
    
    ## Executive Summary
    
    This report covers the performance metrics for January 2024.
    
    ## Key Metrics
    
    - **Total Users**: 1,250
    - **Active Users**: 980 (78.4%)
    - **Revenue**: $45,000
    - **Growth Rate**: 15.2%
    
    ## Highlights
    
    ### User Engagement
    - Average session duration: 12.5 minutes
    - Page views per session: 8.3
    - Bounce rate: 22.1%
    
    ### Revenue
    - Monthly recurring revenue: $45,000
    - New subscriptions: 45
    - Churn rate: 2.1%
    
    ## Recommendations
    
    1. Continue focusing on user engagement initiatives
    2. Implement retention strategies to reduce churn
    3. Explore new revenue streams
    4. Monitor key performance indicators closely
    
    ## Conclusion
    
    The month showed strong performance across all key metrics.
    We recommend maintaining current strategies while exploring
    opportunities for growth.
    """
    
    doc_result = doc_tool.run({
        'content': report_content,
        'document_type': 'pdf',
        'title': 'Monthly Performance Report - January 2024',
        'author': 'SME Agent',
        'subject': 'Performance Analysis',
        'output_path': 'generated_documents/monthly_report_jan_2024.pdf'
    })
    
    print(f"   ✓ {doc_result}")
    
    # Extract file path from result
    file_path = doc_result.split("at: ")[-1] if "at: " in doc_result else 'generated_documents/monthly_report_jan_2024.pdf'
    
    # Step 2: Send email with the report
    print("\n2. Sending email with report attachment...")
    print("   (Note: Requires SMTP configuration)")
    
    # Email body
    email_body = """
    Dear Team,
    
    Please find attached the Monthly Performance Report for January 2024.
    
    Key Highlights:
    - Total Users: 1,250
    - Active Users: 980 (78.4%)
    - Revenue: $45,000
    - Growth Rate: 15.2%
    
    The report contains detailed metrics, analysis, and recommendations.
    Please review and let me know if you have any questions.
    
    Best regards,
    Reporting Team
    """
    
    # HTML version of email body
    html_body = """
    <html>
      <body>
        <h2>Monthly Performance Report - January 2024</h2>
        <p>Dear Team,</p>
        <p>Please find attached the Monthly Performance Report for January 2024.</p>
        
        <h3>Key Highlights:</h3>
        <ul>
          <li><strong>Total Users:</strong> 1,250</li>
          <li><strong>Active Users:</strong> 980 (78.4%)</li>
          <li><strong>Revenue:</strong> $45,000</li>
          <li><strong>Growth Rate:</strong> 15.2%</li>
        </ul>
        
        <p>The report contains detailed metrics, analysis, and recommendations.
        Please review and let me know if you have any questions.</p>
        
        <p>Best regards,<br>Reporting Team</p>
      </body>
    </html>
    """
    
    # Uncomment and configure to actually send email
    """
    email_tool = create_email_automation_tool()
    
    email_result = email_tool.send(
        recipients=['manager@example.com', 'team@example.com'],
        subject='Monthly Performance Report - January 2024',
        body=html_body,
        body_type='html',
        attachments=[file_path],
        cc=['director@example.com']
    )
    
    print(f"   ✓ {email_result}")
    """
    
    print("   (Email sending code is commented out - uncomment and configure SMTP to send)")
    print(f"   Would send to: manager@example.com, team@example.com")
    print(f"   Attachment: {file_path}")
    
    print("\n" + "=" * 60)
    print("Workflow completed!")
    print("=" * 60)


def generate_multiple_formats_and_send():
    """
    Generate report in multiple formats and send all via email.
    """
    print("\n" + "=" * 60)
    print("Generate Multiple Formats & Send")
    print("=" * 60)
    
    doc_tool = create_document_generation_tool()
    
    content = """
    # Quarterly Analysis Report
    
    This report provides a comprehensive analysis of Q1 2024 performance.
    
    ## Summary
    - Revenue: $135,000
    - Users: 3,750
    - Growth: 18.5%
    """
    
    print("\nGenerating reports in multiple formats...")
    
    # Generate PDF
    pdf_path = 'generated_documents/quarterly_report.pdf'
    doc_tool.run({
        'content': content,
        'document_type': 'pdf',
        'title': 'Quarterly Analysis Report',
        'output_path': pdf_path
    })
    print(f"   ✓ Generated: {pdf_path}")
    
    # Generate DOCX
    docx_path = 'generated_documents/quarterly_report.docx'
    doc_tool.run({
        'content': content,
        'document_type': 'docx',
        'title': 'Quarterly Analysis Report',
        'output_path': docx_path
    })
    print(f"   ✓ Generated: {docx_path}")
    
    # Generate PPTX
    pptx_path = 'generated_documents/quarterly_report.pptx'
    doc_tool.run({
        'content': content,
        'document_type': 'pptx',
        'title': 'Quarterly Analysis Report',
        'output_path': pptx_path
    })
    print(f"   ✓ Generated: {pptx_path}")
    
    print("\nSending email with all formats...")
    print("   (Note: Requires SMTP configuration)")
    
    # Uncomment to send
    """
    email_tool = create_email_automation_tool()
    
    email_result = email_tool.send(
        recipients='user@example.com',
        subject='Quarterly Analysis Report - All Formats',
        body='Please find attached the quarterly report in PDF, DOCX, and PPTX formats.',
        attachments=[pdf_path, docx_path, pptx_path]
    )
    
    print(f"   ✓ {email_result}")
    """
    
    print("   (Email sending code is commented out)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DocumentGenerationTool + EmailAutomationTool Integration")
    print("=" * 60)
    
    try:
        generate_and_send_report()
        generate_multiple_formats_and_send()
        
        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)
        print("\nTo actually send emails:")
        print("1. Configure SMTP settings (environment variables or parameters)")
        print("2. Uncomment the email sending code in the examples")
        print("3. Update recipient email addresses")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

