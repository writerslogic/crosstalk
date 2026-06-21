## Description

<!-- Brief description of the changes -->

## Related Issues

<!-- Link to related issues: Fixes #123, Relates to #456 -->

## Type of Change

- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update
- [ ] Refactoring (no functional changes)
- [ ] Security fix

## Checklist

### Code Quality
- [ ] Code follows project style guidelines (`cargo fmt --all -- --check`)
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` is clean
- [ ] Self-review completed
- [ ] No new warnings introduced

### Testing
- [ ] New tests added for changes
- [ ] All existing tests pass (`cargo test --workspace`)
- [ ] Manual testing performed (if applicable)

### Documentation
- [ ] README updated (if needed)
- [ ] CHANGELOG entry added (for user-facing changes)

### Security (for crypto/sandbox/tool-gateway changes)
- [ ] No custom cryptographic primitives introduced
- [ ] Secrets kept in `Zeroizing`; nothing sensitive logged
- [ ] Workspace-confinement / allow-list checks preserved
- [ ] Security implications documented

## Test Evidence

```
$ cargo test --workspace
...
```
