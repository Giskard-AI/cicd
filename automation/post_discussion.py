import huggingface_hub as hf_hub

def construct_post_content(report):
    # Construct the content of the post
    opening = "Hey Team!🤗✨ <br />We’re thrilled to share some amazing evaluation results that’ll make your day!🎉📊<br />"
    disclaimer = "\n\n**Disclaimer**: it's important to note that automated scans may produce false positives or miss certain vulnerabilities. We encourage you to review the findings and assess the impact accordingly.\n"
    whatsnext = "\n### 💡 What's Next?\nThe Giskard community is always buzzing with ideas. 🐢🤔 What do you want to see next? Your feedback is our favorite fuel, so drop your thoughts in the community forum! 🗣️💬 Together, we're building something extraordinary.\n"
    thanks = "\n### 🙌 Big Thanks!\nWe're grateful to have you on this adventure with us. 🚀🌟 Here's to more breakthroughs, laughter, and code magic! 🥂✨ Keep hugging that code and spreading the love! 💻 #Giskard #Huggingface #AISafety 🌈👏 Your enthusiasm, feedback, and contributions are what seek. 🌟 Keep being awesome!\n"

    content = f"{opening}{report}{disclaimer}{whatsnext}{thanks}"
    return content

def create_discussion(repo_id, model_name, hf_token, report):
    description = construct_post_content(report)
    # Create a discussion
    discussion = hf_hub.create_discussion(
        repo_id, 
        title=f"Report for {model_name}", 
        token=hf_token, 
        description=description, 
        repo_type="space")

    return discussion
