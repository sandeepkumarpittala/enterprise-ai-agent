const API_BASE_URL = 'http://127.0.0.1:8000'

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, options)

  const data = await response.json().catch(() => ({}))

  if (!response.ok) {
    const message =
      typeof data.detail === 'string'
        ? data.detail
        : `Request failed with status ${response.status}`

    const error = new Error(message)
    error.status = response.status
    throw error
  }

  return data
}

function authHeaders(token) {
  return {
    Authorization: `Bearer ${token}`,
  }
}

export async function registerUser(email, password) {
  return request('/auth/register', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ email, password }),
  })
}

export async function loginUser(email, password) {
  return request('/auth/login', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ email, password }),
  })
}

export async function getCurrentUser(token) {
  return request('/auth/me', {
    headers: authHeaders(token),
  })
}

export async function uploadDocument(token, file) {
  const formData = new FormData()
  formData.append('file', file)

  return request('/documents/upload', {
    method: 'POST',
    headers: authHeaders(token),
    body: formData,
  })
}

export async function getDocuments(token) {
  return request('/documents', {
    headers: authHeaders(token),
  })
}

export async function deleteDocument(token, documentId) {
  return request(`/documents/${documentId}`, {
    method: 'DELETE',
    headers: authHeaders(token),
  })
}

export async function getConversations(token) {
  return request('/conversations', {
    headers: authHeaders(token),
  })
}

export async function createConversation(token, title = 'New conversation') {
  return request('/conversations', {
    method: 'POST',
    headers: {
      ...authHeaders(token),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ title }),
  })
}

export async function getConversationMessages(token, conversationId) {
  return request(`/conversations/${conversationId}/messages`, {
    headers: authHeaders(token),
  })
}

export async function sendChatMessage(
  token,
  question,
  limit = 5,
  conversationId = null,
) {
  const query = conversationId
    ? `?conversation_id=${encodeURIComponent(conversationId)}`
    : ''

  return request(`/chat${query}`, {
    method: 'POST',
    headers: {
      ...authHeaders(token),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question, limit }),
  })
}
export async function getDocumentChunk(
  token,
  documentId,
  chunkId,
) {
  return request(
    `/documents/${documentId}/chunks/${chunkId}`,
    {
      headers: authHeaders(token),
    },
  )
}