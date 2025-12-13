---
title: Performance Optimization Guide
sidebar_position: 2
description: Techniques and best practices for optimizing the performance of the textbook site
duration: 90
difficulty: advanced
learning_objectives:
  - Implement performance optimization techniques for static sites
  - Optimize images and assets for faster loading
  - Apply code splitting and lazy loading strategies
  - Monitor and measure performance metrics
---

# Performance Optimization Guide

## Learning Objectives

By the end of this guide, you will be able to:
- Apply performance optimization techniques to improve site speed and user experience
- Optimize images and assets for minimal file sizes without quality loss
- Implement code splitting and lazy loading strategies
- Monitor and measure performance metrics
- Troubleshoot common performance bottlenecks

## Performance Fundamentals

### Why Performance Matters

Performance is crucial for the Physical AI & Humanoid Robotics textbook because:

1. **Learning Experience**: Faster loading keeps students engaged and reduces drop-off
2. **Accessibility**: Better performance makes the content accessible to users with slower connections
3. **SEO Benefits**: Search engines favor faster-loading sites
4. **Resource Efficiency**: Optimized sites consume less bandwidth and energy

### Key Performance Metrics

The textbook aims for these performance targets:

- **First Contentful Paint (FCP)**: Under 1.0 second
- **Largest Contentful Paint (LCP)**: Under 2.5 seconds
- **Cumulative Layout Shift (CLS)**: Under 0.1
- **First Input Delay (FID)**: Under 100ms
- **Time to Interactive (TTI)**: Under 3.8 seconds

## Asset Optimization

### Image Optimization

Images are often the largest contributors to page size. Here's how to optimize them:

#### Format Selection

```bash
# Choose the right format for your images
# - JPEG: Photographs and complex images with many colors
# - PNG: Graphics with transparency or few colors
# - WebP: Modern format with superior compression (when browser support is acceptable)
# - SVG: Vector graphics, logos, icons
```

#### Image Compression Tools

Use these tools to compress images:

```bash
# Using ImageOptim (macOS) or Caesium (cross-platform) for batch compression
# Using TinyPNG API for programmatic compression
# Using Squoosh (web-based tool) for fine-tuning compression settings
```

#### Responsive Images

Implement responsive images for different screen sizes:

```html
<!-- Example of responsive image implementation -->
<picture>
  <source media="(max-width: 768px)" srcset="/img/example-mobile.webp" type="image/webp">
  <source media="(max-width: 768px)" srcset="/img/example-mobile.jpg" type="image/jpeg">
  <source media="(min-width: 769px)" srcset="/img/example-desktop.webp" type="image/webp">
  <source media="(min-width: 769px)" srcset="/img/example-desktop.jpg" type="image/jpeg">
  <img src="/img/example-desktop.jpg" alt="Example image" loading="lazy">
</picture>
```

#### Automated Image Optimization

Configure Docusaurus for automatic image optimization:

```typescript
// docusaurus.config.ts
module.exports = {
  themes: [
    [
      '@docusaurus/theme-classic',
      {
        customCss: require.resolve('./src/css/custom.css'),
      },
    ],
  ],
  plugins: [
    [
      '@docusaurus/plugin-client-redirects',
      {
        createRedirects(existingPath) {
          // Create redirects for optimized images
          if (existingPath.includes('/img/')) {
            return [existingPath.replace('/img/', '/img/optimized/')];
          }
        },
      },
    ],
  ],
};
```

### JavaScript Optimization

#### Code Splitting

Docusaurus automatically handles code splitting, but you can enhance it:

```tsx
// Example of manual code splitting for heavy components
import { lazy, Suspense } from 'react';

const HeavyComponent = lazy(() => import('./HeavyComponent'));

function MyPage() {
  return (
    <div>
      <h1>My Page</h1>
      <Suspense fallback={<div>Loading...</div>}>
        <HeavyComponent />
      </Suspense>
    </div>
  );
}
```

#### Bundle Analysis

Analyze your bundle to identify large dependencies:

```bash
# Install webpack bundle analyzer
npm install --save-dev @next/bundle-analyzer

# Add to docusaurus.config.js
const isProd = process.env.NODE_ENV === 'production';

module.exports = {
  // ... rest of config
  plugins: [
    // ... other plugins
    ...(isProd && [
      [
        '@docusaurus/plugin-client-redirects',
        {
          // your prod plugins
        },
      ],
    ]),
  ],
};

# Generate bundle analysis
NODE_ENV=production npm run build
npx serve build
```

### CSS Optimization

#### Critical CSS

Extract and inline critical CSS for faster initial rendering:

```css
/* src/css/critical.css - Critical above-the-fold styles */
.hero {
  display: flex;
  align-items: center;
  padding: 4rem 0;
  min-height: 60vh;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 2rem;
}

/* Non-critical styles go in custom.css */
```

#### CSS Minification and Purging

Configure CSS minification:

```typescript
// docusaurus.config.ts
module.exports = {
  // ... other config
  stylesheets: [
    {
      href: '/css/main.css',
      type: 'text/css',
      crossorigin: undefined,
    },
  ],
};
```

## Loading Strategies

### Lazy Loading

Implement lazy loading for images and components:

```tsx
// LazyLoadImage.tsx
import React, { useState, useEffect, useRef } from 'react';

interface LazyLoadImageProps {
  src: string;
  alt: string;
  className?: string;
  placeholder?: string;
}

const LazyLoadImage: React.FC<LazyLoadImageProps> = ({
  src,
  alt,
  className = '',
  placeholder = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIi8+'
}) => {
  const [isLoaded, setIsLoaded] = useState(false);
  const [isInView, setIsInView] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsInView(true);
          observer.disconnect();
        }
      },
      { threshold: 0.1 }
    );

    if (imgRef.current) {
      observer.observe(imgRef.current);
    }

    return () => {
      if (imgRef.current) {
        observer.unobserve(imgRef.current);
      }
    };
  }, []);

  return (
    <img
      ref={imgRef}
      src={isInView ? src : placeholder}
      alt={alt}
      className={`${className} ${isLoaded ? 'loaded' : 'loading'}`}
      onLoad={() => setIsLoaded(true)}
    />
  );
};

export default LazyLoadImage;
```

### Resource Hints

Use resource hints to preload critical resources:

```html
<!-- In your site's head -->
<link rel="preload" href="/fonts/main-font.woff2" as="font" type="font/woff2" crossorigin>
<link rel="prefetch" href="/api/content.json">
<link rel="dns-prefetch" href="//analytics.example.com">
```

## Caching Strategies

### Browser Caching

Configure proper caching headers:

```javascript
// For GitHub Pages or CDN configuration
// Cache static assets for longer periods
// Cache HTML files for shorter periods to pick up changes quickly

// Example for nginx configuration
/*
location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
}

location ~* \.html$ {
    expires 1h;
    add_header Cache-Control "public, must-revalidate";
}
*/
```

### Service Worker Caching

Implement service worker for offline capability:

```javascript
// sw.js - Simple service worker for caching
const CACHE_NAME = 'textbook-v1';
const urlsToCache = [
  '/',
  '/css/main.css',
  '/js/main.js',
  '/offline.html'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // Return cached version if available
        if (response) {
          return response;
        }
        // Otherwise fetch from network
        return fetch(event.request);
      })
  );
});
```

## Performance Monitoring

### Core Web Vitals Monitoring

Set up monitoring for Core Web Vitals:

```typescript
// src/utils/web-vitals.ts
import { getCLS, getFID, getFCP, getLCP, getTTFB } from 'web-vitals';

const sendToAnalytics = ({ name, value, id }: { name: string; value: number; id: string }) => {
  // Replace with your analytics endpoint
  (window as any).gtag?.('event', name, {
    event_category: 'Web Vitals',
    event_label: id,
    value: Math.round(name === 'CLS' ? value * 1000 : value),
    non_interaction: true,
  });
};

// Measure and send web vitals
getCLS(sendToAnalytics);
getFID(sendToAnalytics);
getFCP(sendToAnalytics);
getLCP(sendToAnalytics);
getTTFB(sendToAnalytics);
```

### Performance Budget

Create a performance budget to prevent regressions:

```json
// performance-budget.json
{
  "resourceSizes": [
    {
      "resourceType": "document",
      "budget": 100
    },
    {
      "resourceType": "script",
      "budget": 300
    },
    {
      "resourceType": "stylesheet",
      "budget": 50
    },
    {
      "resourceType": "image",
      "budget": 1000
    },
    {
      "resourceType": "font",
      "budget": 150
    }
  ],
  "timings": [
    {
      "metric": "firstContentfulPaint",
      "budget": 1000
    },
    {
      "metric": "largestContentfulPaint",
      "budget": 2500
    },
    {
      "metric": "cumulativeLayoutShift",
      "budget": 0.1
    }
  ]
}
```

### Performance Testing

Automate performance testing in your CI/CD pipeline:

```yaml
# .github/workflows/performance.yml
name: Performance Testing
on: [pull_request]

jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Node.js
        uses: actions/setup-node@v2
        with:
          node-version: '18'

      - name: Install dependencies
        run: |
          cd docs
          npm ci

      - name: Build site
        run: |
          cd docs
          npm run build

      - name: Run Lighthouse CI
        run: |
          npm install -g @lhci/cli@0.12.x
          lhci autorun
        env:
          LHCI_GITHUB_APP_TOKEN: ${{ secrets.LHCI_GITHUB_APP_TOKEN }}
```

## Advanced Optimization Techniques

### Tree Shaking

Remove unused code from bundles:

```javascript
// Use ES6 imports to enable tree shaking
import { specificFunction } from 'library'; // Good - only imports what's needed
import * as Library from 'library'; // Avoid - imports everything

// In package.json, mark side-effect-free modules
{
  "sideEffects": false
}
```

### Resource Compression

Enable compression on your server:

```bash
# For gzip compression
npm install compression-webpack-plugin

# In webpack config
const CompressionPlugin = require('compression-webpack-plugin');

module.exports = {
  plugins: [
    new CompressionPlugin({
      algorithm: 'gzip',
      test: /\.(js|css|html|svg)$/,
      threshold: 8192,
      minRatio: 0.8,
    }),
  ],
};
```

### Font Optimization

Optimize web fonts for performance:

```css
/* Use font-display: swap to prevent invisible text during font loading */
@font-face {
  font-family: 'Roboto';
  src: url('/fonts/Roboto-Regular.woff2') format('woff2');
  font-display: swap; /* Shows fallback font immediately, swaps to web font when loaded */
}

/* Preload critical fonts */
/* Add to HTML head: <link rel="preload" href="/fonts/Roboto-Regular.woff2" as="font" type="font/woff2" crossorigin> */
```

## Mobile Optimization

### Responsive Design Performance

Optimize for mobile devices:

```css
/* Use responsive units and optimize for mobile first */
@media (max-width: 768px) {
  .container {
    padding: 0 1rem; /* Reduce padding on mobile */
  }

  /* Hide non-critical elements on small screens */
  .desktop-only {
    display: none;
  }
}

/* Use contain property to limit style and layout recalculation */
.widget {
  contain: layout style paint;
}
```

### Touch and Interaction Optimization

Optimize for touch interfaces:

```css
/* Ensure touch targets are large enough */
.button, .link {
  min-height: 44px; /* Minimum recommended touch target size */
  min-width: 44px;
  padding: 12px;
}

/* Optimize scrolling performance */
.scroll-container {
  scroll-behavior: smooth;
  -webkit-overflow-scrolling: touch; /* Enable momentum scrolling on iOS */
}
```

## Performance Auditing

### Lighthouse Audit

Run regular Lighthouse audits:

```bash
# Install Lighthouse CLI
npm install -g lighthouse

# Run audit
lighthouse http://localhost:3000 --output html --output-path ./report.html

# For CI/CD integration
npm install -g @lhci/cli
lhci collect --url=https://yoursite.com
lhci assert --assertions.chromeWebstoreRating="off" --assertions.first-contentful-paint.warn=600
```

### WebPageTest Analysis

Use WebPageTest for detailed analysis:

```bash
# WebPageTest can be used to analyze:
# - Waterfall charts showing resource loading
# - Performance comparison across different connection speeds
# - Geographic performance variations
```

## Troubleshooting Performance Issues

### Common Performance Bottlenecks

1. **Large JavaScript bundles**: Use bundle analysis to identify culprits
2. **Unoptimized images**: Implement proper compression and responsive serving
3. **Render-blocking resources**: Defer non-critical CSS and JS
4. **Third-party scripts**: Audit and optimize external dependencies
5. **Layout thrashing**: Batch DOM reads and writes

### Debugging Tools

Use browser dev tools for performance debugging:

```javascript
// Performance measurement in code
const measurePerformance = (callback, name) => {
  const start = performance.now();
  callback();
  const end = performance.now();
  console.log(`${name} took ${end - start} milliseconds`);
};

measurePerformance(() => {
  // Code to measure
}, 'Critical Path Function');
```

## Performance Maintenance

### Continuous Monitoring

Set up continuous performance monitoring:

1. **Regular audits**: Schedule weekly/monthly performance audits
2. **Alerts**: Set up alerts for performance regressions
3. **Documentation**: Keep performance guidelines updated
4. **Team awareness**: Educate team on performance best practices

### Performance Culture

Foster a performance-first culture:

- Include performance in code reviews
- Set performance goals for new features
- Celebrate performance improvements
- Share performance wins with the team

By implementing these optimization techniques, the Physical AI & Humanoid Robotics textbook will provide an excellent learning experience with fast loading times and smooth interactions, regardless of the user's device or connection speed.