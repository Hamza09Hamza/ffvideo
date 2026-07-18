export async function postVerify({ serverUrl, apiKey, frameBlobs }) {
  const form = new FormData()
  frameBlobs.forEach((blob, i) => form.append('frames', blob, `frame_${i}.jpg`))

  const res = await fetch(`${serverUrl}/api/v1/verify`, {
    method: 'POST',
    headers: { 'X-API-Key': apiKey },
    body: form,
  })
  if (!res.ok) throw new Error(`Server error: ${res.status}`)
  return res.json()
}

export async function postEnroll({ serverUrl, apiKey, fullName, email, poseBlobs }) {
  const form = new FormData()
  form.append('full_name', fullName)
  if (email) form.append('email', email)
  for (const [pose, blob] of Object.entries(poseBlobs)) {
    form.append(pose, blob, `${pose}.jpg`)
  }

  const res = await fetch(`${serverUrl}/api/v1/enroll`, {
    method: 'POST',
    headers: { 'X-API-Key': apiKey },
    body: form,
  })
  if (!res.ok) throw new Error(`Server error: ${res.status}`)
  return res.json()
}
