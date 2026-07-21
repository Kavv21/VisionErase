# AUTH_SSO.md correction (read alongside the original)

The original `docs/AUTH_SSO.md` Section 5 config note says to run:

```sql
alter database postgres set app.settings.allowed_email_domain = 'ahduni.edu.in';
```

**This does not work on Supabase's managed Postgres** — the project's
connection role doesn't have permission to set custom GUC namespaces at
the database level (`permission denied to set parameter`), even though
this works on a self-hosted Postgres instance.

Fixed in `supabase/migrations/0003_app_config_and_domain_fix.sql`: the
allowed domain now lives in a real table (`app_config`), which
`handle_new_user()` reads instead of `current_setting()`. To change the
domain later, update the row rather than editing database settings:

```sql
update app_config set value = 'new-domain.example', updated_at = now()
where key = 'allowed_email_domain';
```
