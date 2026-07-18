# Face ID: Database & Multi-App Auth Design

## The core decision: one centralized database, not one per app

Every app inside the company that needs to know "who is this person" (admin
panels, internal tools, whatever comes later) talks to **the same** Face ID
backend and **the same** Postgres database. No app keeps its own copy of
enrolled faces.

Why this matters:

- **One enrollment, usable everywhere.** An employee enrolls their face
  once. Every app that trusts this service can recognize them immediately —
  nobody re-enrolls per app.
- **One place to revoke/update.** If someone leaves the company, deleting
  their `employees` row removes their access everywhere at once (cascades
  to their embeddings automatically — see below). With one DB per app,
  you'd have to remember to do that N times.
- **Biometric data is sensitive.** Fewer places storing face embeddings
  means a smaller, easier-to-audit surface for something this sensitive,
  instead of it being scattered across every app's own storage.

What each app *doesn't* need to duplicate: the database, the enrollment
logic, the matching logic, the anti-spoofing model. What each app *does*
still own: its own camera-capture UI (native to whatever platform it's
built on — see `face_id/web/` for the reference implementation) and
whatever it does with the identity once it gets one back.

---

## The schema, table by table

Defined in `face_id/server/schema.sql`.

### `employees`

```sql
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    full_name TEXT NOT NULL,
    email TEXT UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

- `id SERIAL PRIMARY KEY` — an auto-incrementing integer, the stable
  identifier every other table (and every JWT — see below) refers to. Never
  reuse or repurpose this value; it's the one thing that never changes
  even if someone's name or email does.
- `email TEXT UNIQUE` — enforced uniqueness at the database level, not just
  in application code. This is why re-enrolling the same email crashes
  with `UniqueViolation` right now (a real constraint doing its job) —
  see the "known gap" note below about handling that gracefully instead of
  crashing.
- `created_at` — every table like this should record when a row was
  created. Costs nothing, and you cannot get this back later if you didn't
  capture it at the time.

### `face_embeddings`

```sql
CREATE TABLE face_embeddings (
    id SERIAL PRIMARY KEY,
    employee_id INTEGER NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    embedding VECTOR(512) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

- **Why a separate table instead of a column on `employees`:** one
  employee has *several* embeddings (one per enrolled pose — center, left,
  right, up, down; see `face_id/server/enrollment.py`). A one-to-many
  relationship like this always gets its own table in relational design —
  cramming a variable-length list into a single row's column is the
  classic sign a table should be split out.
- `employee_id ... REFERENCES employees(id) ON DELETE CASCADE` — this is a
  **foreign key**: the database itself refuses to let a `face_embeddings`
  row point at an `employee_id` that doesn't exist in `employees`. Real
  data integrity, enforced by Postgres, not just hoped for by application
  code. `ON DELETE CASCADE` means: delete an employee, and all their
  embeddings vanish automatically — you never end up with orphaned
  embedding rows pointing at a deleted person.
- `VECTOR(512)` — this type comes from the `pgvector` extension
  (`CREATE EXTENSION vector;`, already enabled on the `face_id` database).
  ArcFace (the recognition model) always outputs exactly 512 numbers per
  face, so the column is fixed-width, not a generic array.
- **No approximate-search index (ivfflat/hnsw)** — deliberately. Those
  trade a little accuracy for speed on huge datasets (think: millions of
  rows). At the target scale here (~200-300 employees × 5 poses ≈ 1,000-
  1,500 rows), a plain sequential scan with pgvector's `<=>` cosine
  distance operator is both faster to set up and *exact*, no approximation
  error. Revisit this only if the row count grows by orders of magnitude.

### How matching actually reads this schema (`face_id/server/db.py`)

```sql
SELECT e.id, e.full_name, 1 - (fe.embedding <=> %s) AS similarity
FROM face_embeddings fe
JOIN employees e ON e.id = fe.employee_id
ORDER BY fe.embedding <=> %s
LIMIT 1
```

This is a 1-nearest-neighbor search across **every embedding of every
employee at once** — not "the average embedding per person." Whichever
single stored embedding (any pose, any employee) is closest to the live
face wins. That's why enrolling multiple distinct poses matters: more
reference points per person means whatever angle they show up at later,
something close is likely already stored.

---

## The auth/token flow — how other apps actually use this

This is the part that makes "just push to an API and it works" true for
things like admin panels.

1. An app (any platform) captures a face, POSTs it to
   `POST /api/v1/verify` with its `X-API-Key` header.
2. On a successful match, the response includes a **signed JWT**:
   ```json
   {
     "success": true,
     "employee_id": 5,
     "full_name": "Hamza Boukader",
     "similarity": 0.61,
     "token": "eyJhbGciOiJIUzI1NiIs..."
   }
   ```
3. That JWT encodes `employee_id`, `full_name`, an issued-at time, and a
   **5-minute expiry** — signed with `JWT_SECRET` (in `.env`, shared only
   between trusted backend services, never sent to a browser/client).
4. Any app that needs to confirm "this really is Hamza Boukader, verified
   by face, just now" can **verify the JWT's signature itself**, offline,
   using the shared secret — no network call back to this server required:
   ```python
   import jwt
   payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
   # payload["employee_id"], payload["full_name"] are now trustworthy
   ```
   If the signature doesn't match, or the token's expired, `jwt.decode`
   raises — the app knows immediately not to trust it.

**Why 5 minutes, not a long-lived session:** this token proves "verified
by face just now," not "logged in for the day." A consuming app (say, an
admin panel) should exchange it immediately for whatever session mechanism
it already uses internally (its own cookie, its own longer session token)
rather than passing this JWT around as a standing credential. Short TTL
limits the damage if a token were ever intercepted.

**What this doesn't do yet — revocation.** A stateless JWT is valid until
it expires, full stop; there's no way to invalidate one early (e.g. someone
walks out mid-token-lifetime and you want to kill their access
immediately). At a 5-minute TTL this is a small window, so it's not
currently handled — if it ever needs to be, the fix is a small
`revoked_tokens` table (store a token's unique ID and check it on
decode), not a redesign.

---

## Onboarding a new app

1. **Generate it an API key.** Right now there's a single shared `API_KEY`
   in `.env` — fine for one or two apps. Once several distinct apps are
   calling in, worth moving to a small `api_clients` table (`id`,
   `name`, `key_hash`, `created_at`) so each app has its own key, and a
   compromised or retired app's key can be revoked individually instead of
   rotating one shared secret for everyone. Not built yet — flagging it as
   the natural next step, not a missing requirement today.
2. **Give it `JWT_SECRET`** (or, if it only ever *receives* tokens from
   users who already verified elsewhere and never calls `/verify` itself,
   it only needs the secret to *decode*, not the API key at all).
3. The app builds its own camera capture (see `face_id/web/` for the
   reference React implementation) and calls `/api/v1/verify` /
   `/api/v1/enroll` directly. No new backend code, no new database.

---

## Adding a new table later (worked example)

Say you eventually want to track *which app* verified someone and when,
for an audit log. The pattern to follow:

```sql
CREATE TABLE verification_log (
    id SERIAL PRIMARY KEY,
    employee_id INTEGER NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    app_name TEXT NOT NULL,
    similarity REAL NOT NULL,
    verified_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ON verification_log(employee_id);
```

The same reasoning as above applies every time: does this data belong to
exactly one employee, or many (one-to-many → separate table, foreign key
back to `employees`)? Does it need a stable ID other tables might reference
later? Should deleting the employee delete this too (`CASCADE`), or should
the record survive for historical audit purposes (in which case, don't
cascade — leave the row, maybe with `employee_id` nullable once they're
gone)? That last question is exactly the kind of judgment call worth
making explicitly, not by default.

Apply new tables the same way the current schema was applied:

```bash
PGPASSWORD='...' psql -h localhost -U face_id_app -d face_id -f face_id/server/schema.sql
```

(`CREATE TABLE IF NOT EXISTS` in the existing schema means re-running it is
always safe — it won't touch tables that already exist.)

---

## Known gap: re-enrolling an existing email

Right now, POSTing to `/api/v1/enroll` with an email that's already
enrolled crashes with a raw Postgres `UniqueViolation` (a 500 error) rather
than a clean response. The constraint is doing its job (preventing
duplicate identities), but the failure mode is ugly. Worth fixing —
either a clean `{"success": false, "reason": "already_enrolled"}` response,
or actual support for re-enrollment (replacing an employee's embeddings)
depending on whether that's a real use case (e.g. re-enrolling after a
haircut/beard/glasses change significantly alters someone's face).
