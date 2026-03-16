import express from 'express'
import cors from 'cors'
import dotenv from 'dotenv'
import Groq from 'groq-sdk'

dotenv.config()

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY })

const app = express()
app.use(cors())
app.use(express.json())

app.post('/api/chat', async (req, res) => {
  try {
    const { system, messages } = req.body

    const response = await groq.chat.completions.create({
      model: 'llama-3.1-8b-instant',
      messages: [
        { role: 'system', content: system },
        ...messages.map(m => ({ role: m.role, content: m.content }))
      ]
    })

    const text = response.choices[0].message.content
    res.json({ content: [{ text }] })
  } catch (err) {
    console.error('API Error:', err.message)
    res.status(500).json({ error: err.message })
  }
})

app.listen(3001, () => console.log('Server running on http://localhost:3001'))