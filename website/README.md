# MASK-Bench Website

This is the website for MASK-Bench (Multimodal Assessment of Skills and Knowledge Benchmark).

## GitHub Pages Deployment

To deploy this website on GitHub Pages:

1. Push this `website` directory to your GitHub repository
2. Go to your repository Settings → Pages
3. Under "Source", select "Deploy from a branch"
4. Choose the branch containing the website directory
5. Set the folder to `/website`
6. Click "Save"

The site will be available at: `https://[username].github.io/[repository-name]/`

## Local Development

To view the website locally:

1. Open `index.html` in a web browser
2. Or use a local server:
   ```bash
   cd website
   python -m http.server 8000
   ```
   Then visit `http://localhost:8000`

## Structure

- `index.html` - Main page content
- `styles.css` - Styling and responsive design
- `images/` - Directory for images (add your benchmark overview and dataset distribution images here)
- `.nojekyll` - Tells GitHub Pages to not process files with Jekyll

## Images Needed

Please add the following images to the `images/` directory:
- `benchmark-overview.png` - Overview diagram of MASK-Bench
- `dataset-distribution.png` - Chart showing dataset distribution

## Customization

Update the following in `index.html`:
- Paper link (currently placeholder)
- Hugging Face dataset link
- GitHub repository link
- Contact email
- Citation details