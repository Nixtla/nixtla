## Introduction

This repository contains the documentation for Nixtla TimeGPT.
It is the third iteration of the docs:

- First version was hosted in readme.com and still visible there for reference. [https://timegpt.readme.io/](https://timegpt.readme.io/) and in this same doc in the [migration to mintlify](#migration-from-readmeio-to-mintlify) section you can see some details about the migration.

- Second version has exactly this same content but it's in the same repo as the website, which isn't ideal for collaborators, currently visible at [https://nixtla.io/docs/](https://nixtla.io/docs/) and hosted in mintlify.

- This is iteration will also be hosted in mintlify.

## Contributing

### Option 1: Using Mintlify Online Editor

The easiest and safest way to make changes to the documentation is using the [online mintlify editor](https://dashboard.mintlify.com/nixtla/nixtla-docs/editor/main).

The mintlify editor will send a PR in github so the team can review the changes and merge them. Same process as if you were contributing to the repo in any other way.

### Option 2: Using GitHub Codespaces

If you prefer to work locally, this repository is configured with GitHub Codespaces to easily view and develop documentation with Mintlify.

1. Click the "Code" button at the top of this repository
2. Select the "Codespaces" tab
3. Click "Create codespace on [branch]"
4. Once the Codespace is ready, open a terminal and run:

   ```bash
   mintlify dev
   ```

5. Click on the "Ports" tab and open port 3000 in your browser to view the documentation

For more information about the Codespace setup, see the [.devcontainer](/.devcontainer) directory.

Please be aware that linters and formatters fail most of the time because the docs follow a very specific structure that is not supported by the default linters. So please turn them off when working locally to prevent extra changes in your PRs like adding or removing spaces in the code.

## Release process

When a PR is merged to main branch, the docs are deployed to production automatically.

## Pending improvements and wishlist

### Technical

- [ ] Add support for .ipynb files, so that we can run the notebooks and output the results in the docs.
- [ ] Implement Nixtla branding on the docs.
- [ ] Organize the docs in subfolders.

### Content

- [ ] Rethink how we welcome non technical users into the docs, currently we start the into with "Welcome to TimeGPT - The foundational model for time series forecasting and anomaly detection" which sounds very overwhelming for a non technical user. It would be nice to add content for people who are just starting to work with time series forecasting and anomaly detection.
- [ ] ????

## Migration from Readme.io to Mintlify

#### HTML to Markdown Link Conversion

We've standardized the documentation by converting all HTML links (`<a href="URL">text</a>`) to Markdown format (`[text](URL)`). This makes the documentation more consistent and easier to maintain.

#### Conversion Script

The conversion was done using a Node.js script located at `/docs/utils/convert-links.js`. To run the script:

1. Install dependencies (if not already installed):

   ```bash
   npm install glob
   ```

2. Change to the docs directory:

   ```bash
   cd docs
   ```

3. Run the script:

   ```bash
   node utils/convert-links.js
   ```

For testing a single file without making changes:

```bash
node utils/convert-links.js --test path/to/file.mdx
```

For a dry run (simulate changes without writing to files):

```bash
node utils/convert-links.js --dry-run
```

### HTML to Markdown General Conversion

We've enhanced our documentation by converting various HTML elements to their Markdown equivalents for improved readability and maintainability. This comprehensive conversion captures elements beyond just links.

#### Conversion Script

The conversion was performed using a more advanced Node.js script at `/docs/utils/html-to-markdown.js`. This script:

1. Converts HTML formatting to Markdown equivalents
2. Preserves component structure and attributes
3. Makes the content more consistent with Markdown best practices

To run the script:

1. Change to the docs directory: `cd docs`

2. Run the script:

```bash
node utils/html-to-markdown.js
```

For testing a single file without making changes:

```bash
node utils/html-to-markdown.js --test path/to/file.mdx
```

For a dry run (simulate changes without writing to files):

```bash
node utils/html-to-markdown.js --dry-run
```

### Frame Component Image Standardization

We've standardized how images are used inside Frame components by ensuring all images use Markdown syntax (`![alt](url)`) rather than HTML `img` tags or the deprecated `src` attribute.

#### Frame Image Fix Script

The standardization was done using a Node.js script located at `/docs/utils/fix-frame-images.js`. This script fixes two issues:

1. Converts `<Frame src="URL" alt="ALT"></Frame>` to `<Frame>![ALT](URL)</Frame>`
2. Converts `<Frame><img src="URL" alt="ALT" /></Frame>` to `<Frame>![ALT](URL)</Frame>`

To run the script:

1. Change to the docs directory:

   ```
   cd docs
   ```

2. Run the script:

   ```
   node utils/fix-frame-images.js
   ```

For testing a single file without making changes:

```
node utils/fix-frame-images.js --test path/to/file.mdx
```

For a dry run (simulate changes without writing to files):

```
node utils/fix-frame-images.js --dry-run
```

### Card Title Standardization

We've standardized how titles are defined in Card components by ensuring they use the `title` prop rather than bold text (`**Text**`) inside the card body.

#### Card Title Check Script

To identify and fix cards with incorrectly defined titles, use the script at `/docs/utils/check-card-titles.js`. This script:

1. Identifies Card components where the title is defined using bold text instead of the `title` prop
2. Reports which files have this issue and shows what needs to be fixed
3. Can automatically fix the issues with the `--fix` flag

To run the script:

1. Change to the docs directory:

   ```
   cd docs
   ```

2. Check for issues without making changes:

   ```
   node utils/check-card-titles.js
   ```

3. Automatically fix issues:

   ```
   node utils/check-card-titles.js --fix
   ```

For testing a single file:

```
node utils/check-card-titles.js --test path/to/file.mdx
```

For a dry run (simulate fixes without making changes):

```
node utils/check-card-titles.js --fix --dry-run
```

### Table Conversion to Markdown

We've standardized how tables are presented in documentation by converting all `<Table>` components to native Markdown table syntax. This makes the documentation more consistent and easier to maintain.

#### Table Conversion Script

To convert tables from HTML to Markdown format, use the script at `/docs/utils/convert-tables.js`. This script:

1. Identifies `<Table>` components in MDX files
2. Converts them to Markdown table syntax (`| Header | Header |` format)
3. Preserves all table data while simplifying the format

To run the script:

1. Change to the docs directory:

   ```
   cd docs
   ```

2. Check for tables without making changes:

   ```
   node utils/convert-tables.js
   ```

3. Automatically convert all tables:

   ```
   node utils/convert-tables.js --fix
   ```

For testing a single file:

```
node utils/convert-tables.js --test path/to/file.mdx
```

For a dry run (simulate conversions without making changes):

```
node utils/convert-tables.js --fix --dry-run
```

### Publishing Changes

Install our Github App to auto propagate changes from your repo to your deployment. Changes will be deployed to production automatically after pushing to the default branch. Find the link to install on your dashboard.

#### Troubleshooting

- Mintlify dev isn't running - Run `mintlify install` it'll re-install dependencies.
- Page loads as a 404 - Make sure you are running in a folder with `docs.json`
