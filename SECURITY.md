# Security and privacy reporting

Do not open a public issue containing clinical data, credentials, internal infrastructure details, screenshots of private records, model weights, adapters, checkpoints, optimizer states, interpretability matrices, Git LFS objects, or links to private model artifacts.

If sensitive material is discovered in Git history, immediately:

1. Restrict repository visibility and revoke any exposed credential.
2. Notify the institution's security/privacy contact through an approved private channel.
3. Remove the material from the complete Git history; deleting the latest file is insufficient.
4. Treat the data as disclosed until caches, forks and mirrors have been assessed.

This repository intentionally provides no clinical-data upload mechanism.

The project also intentionally provides no model-artifact release channel. The same prohibition applies to issues, pull requests, discussions, releases and mirrors presented as part of this project.
