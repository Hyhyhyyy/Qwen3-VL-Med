# Data governance and internal encryption

## Public-release rule

The public repository is code-only. Real images, reports, row-level outputs, cohort manifests, identifiers, model weights trained on restricted data, and private infrastructure metadata are excluded rather than anonymized in place.

De-identification alone is not treated as authorization to publish. Free text and pathology images may retain indirect identifiers or rare combinations that enable re-identification.

## Internal storage

Restricted artifacts should remain on institution-managed encrypted storage with access logging, least-privilege permissions, retention limits and a documented data owner. Encryption keys must be stored separately from ciphertext using an approved password manager or key-management service.

For an authorized offline transfer, an institution-approved tool may create an AES-256 encrypted archive with encrypted filenames. One example using 7-Zip is:

```powershell
7z a -t7z -mhe=on -p protected-internal.7z D:\approved\restricted-artifacts
```

The command prompts for a passphrase. Never place the passphrase in a command, script, environment file, Git commit or chat. Verify the archive can be decrypted in the approved destination before deleting any source. Do not upload the encrypted archive to this repository.

The repository also provides `tools/create_private_archive.ps1`, which refuses to write the archive inside the repository, prompts for the passphrase through 7-Zip, verifies decryption integrity, and writes a separate SHA-256 sidecar. It does not authorize transfer or replace institutional approval.

## Required release gate

Before every push:

1. Build from an explicit allowlist, not by copying the private project tree.
2. Run `tools/privacy_audit.ps1`.
3. Run `python tools/privacy_audit_history.py`.
4. Review `git diff --cached` and `git status` manually.
5. Confirm no large files or Git LFS objects are present.
6. Confirm Git history contains only sanitized commits.
