# Owner-Facing Pages

The owner requires every new local page, case, notebook export and report to be
reachable from its corresponding section of the project home page:
`http://127.0.0.1:8765/framework.html`.

- Do not deliver orphan `file://` pages or rely on a separate temporary port.
- Register notebook HTML exports in `studio/src/framework/resources.js` under
  the correct domain and, for cases, the correct direction. Existing domain
  and direction identifiers are in `studio/src/framework/data.js`.
- Use the shared local server at `studio/scripts/serve.mjs`. Keep return-to-home
  navigation and notebook downloads working; preserve original experiment
  outputs, accuracy caveats and hardware receipts.
- Validate registration and links with the navigation checks documented in
  `studio/AGENTS.md`. Test actual home-to-page-to-home navigation on desktop
  and mobile before delivering a page.
