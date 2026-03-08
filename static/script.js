document.getElementById('chatbot-icon').addEventListener('click', () => {

    const chatWindow = document.getElementById('chat-window');
    
    chatWindow.style.display = chatWindow.style.display === 'none' ? 'flex' : 'none';
    
    });
    
    
    // Handle message sending
    document.getElementById('send-button').addEventListener('click', async () => {
    
    const userMessage = document.getElementById('user-message').value;
    
    if(!userMessage) return;
    
    // add user message
    const chatMessages = document.getElementById('chat-message');
    
    const userBubble = document.createElement('div');
    
    userBubble.textContent = `You: ${userMessage}`;
    
    chatMessages.appendChild(userBubble);
    // Call  backend API

    const response = await fetch('/chat',{
        method:'POST',
        headers: {'Content-Type':'application/json'},
        body :JSON.stringify({message:userMessage})


    });
    const data = await response.json()

    //Add bot 's response 
    const botBubble = document.createElement('div');
    botBubble.textContent = `Bot:${data.response}`;
    chatMessages.appendChild(botBubble);

    //Scroll to bottom
    chatMessages.scrollTop=chatMessages.scrollHeight;

    // clear input 
    document.getElementById('user-message').value='';

    });