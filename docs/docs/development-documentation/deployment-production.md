---
title: Deployment and Production Configuration
sidebar_position: 5
description: Complete guide for deploying and maintaining the Physical AI & Humanoid Robotics textbook in production
duration: 150
difficulty: advanced
learning_objectives:
  - Configure production deployment for the textbook
  - Set up monitoring and analytics for production
  - Implement security best practices
  - Optimize for performance and scalability
  - Create rollback and disaster recovery procedures
---

# Deployment and Production Configuration

## Learning Objectives

By the end of this guide, you will be able to:
- Configure and deploy the Physical AI & Humanoid Robotics textbook to production
- Set up monitoring and analytics for the production environment
- Implement security best practices for educational content
- Optimize the site for performance and scalability
- Create rollback and disaster recovery procedures
- Maintain the production environment effectively

## Production Architecture

### Infrastructure Overview

The Physical AI & Humanoid Robotics textbook is designed for deployment using static site hosting with CDN distribution:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GitHub      │───▶│  CI/CD Pipeline │───▶│  Static Hosting │
│   Repository  │    │   (GitHub Actions)│    │   (Netlify/    │
└─────────────────┘    └─────────────────┘    │   Vercel/      │
                                              │   GitHub Pages) │
                                              └─────────────────┘
                                                       │
                                                       ▼
                                               ┌─────────────────┐
                                               │   CDN Layer     │
                                               │   (Cloudflare)  │
                                               └─────────────────┘
                                                       │
                                                       ▼
                                               ┌─────────────────┐
                                               │   End Users     │
                                               └─────────────────┘
```

### Technology Stack

The production deployment uses:

- **Static Site Generator**: Docusaurus 3.x
- **Frontend Framework**: React with TypeScript
- **Hosting**: GitHub Pages, Netlify, or Vercel
- **CDN**: Cloudflare (recommended)
- **Analytics**: Google Analytics or Plausible
- **Monitoring**: Custom performance monitoring
- **Security**: HTTPS, CSP, security headers

## Production Configuration

### Docusaurus Production Configuration

Configure Docusaurus for production deployment:

```typescript
// docusaurus.config.prod.ts
import { Config } from '@docusaurus/types';
import { themes as prismThemes } from 'prism-react-renderer';

const config: Config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'Comprehensive textbook for industry engineers',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://your-domain.com',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub Pages: https://username.github.io/repo-name/
  baseUrl: '/',

  // GitHub pages deployment config
  organizationName: 'organization-name',
  projectName: 'physical-ai-textbook',
  trailingSlash: false,

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'es', 'fr'], // Add more as needed
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        blog: false, // Disable blog if not needed
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
    navbar: {
      title: 'Physical AI & Humanoid Robotics',
      logo: {
        alt: 'Physical AI Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'textbook',
          position: 'left',
          label: 'Textbook',
        },
        {
          href: 'https://github.com/your-org/physical-ai-textbook',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Content',
          items: [
            {
              label: 'Textbook',
              to: '/docs/category/week-1-2-introduction',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Stack Overflow',
              href: 'https://stackoverflow.com/questions/tagged/docusaurus',
            },
            {
              label: 'Discord',
              href: 'https://discordapp.com/invite/docusaurus',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/facebook/docusaurus',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'json', 'yaml', 'docker'],
    },
  } satisfies Preset.ThemeConfig,

  plugins: [
    // Plugin for sitemap generation
    [
      '@docusaurus/plugin-sitemap',
      {
        changefreq: 'weekly',
        priority: 0.5,
        ignorePatterns: ['/tags/**'],
        filename: 'sitemap.xml',
      },
    ],
    // Plugin for Google Analytics
    [
      '@docusaurus/plugin-google-analytics',
      {
        trackingID: 'GA_TRACKING_ID',
        anonymizeIP: true,
      },
    ],
    // Plugin for PWA support
    [
      '@docusaurus/plugin-pwa',
      {
        debug: false,
        offlineModeActivationStrategies: [
          'appInstalled',
          'standalone',
          'queryString',
        ],
        pwaHead: [
          {
            tagName: 'link',
            rel: 'icon',
            href: '/img/logo.svg',
          },
          {
            tagName: 'link',
            rel: 'manifest',
            href: '/manifest.json',
          },
          {
            tagName: 'meta',
            name: 'theme-color',
            content: 'rgb(37, 194, 160)',
          },
          {
            tagName: 'meta',
            name: 'apple-mobile-web-app-capable',
            content: 'yes',
          },
          {
            tagName: 'meta',
            name: 'apple-mobile-web-app-status-bar-style',
            content: '#000',
          },
          {
            tagName: 'link',
            rel: 'apple-touch-icon',
            href: '/img/apple-touch-icon.png',
          },
          {
            tagName: 'link',
            rel: 'mask-icon',
            href: '/img/logo.svg',
            color: 'rgb(62, 204, 94)',
          },
          {
            tagName: 'meta',
            name: 'msapplication-TileImage',
            content: '/img/logo.svg',
          },
          {
            tagName: 'meta',
            name: 'msapplication-TileColor',
            content: '#000',
          },
        ],
      },
    ],
  ],
};

export default config;
```

### Production Environment Variables

Create production environment configuration:

```bash
# .env.production
# Production environment variables
NODE_ENV=production

# Analytics
GA_TRACKING_ID=UA-XXXXXXXXX-X

# API endpoints (if any backend services are needed)
REACT_APP_API_BASE_URL=https://api.your-domain.com

# Feature flags
REACT_APP_FEATURE_NEW_NAVIGATION=true
REACT_APP_FEATURE_ADVANCED_SEARCH=true

# CDN configuration
REACT_APP_CDN_URL=https://cdn.your-domain.com

# Security
REACT_APP_CONTENT_SECURITY_POLICY="default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://www.google-analytics.com; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' https://fonts.gstatic.com; connect-src 'self' https://www.google-analytics.com;"
```

## Build and Deployment Process

### Production Build Configuration

Configure the production build process:

```json
// docs/package.json
{
  "name": "physical-ai-textbook",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "docusaurus": "docusaurus",
    "start": "docusaurus start",
    "build": "docusaurus build",
    "swizzle": "docusaurus swizzle",
    "deploy": "docusaurus deploy",
    "clear": "docusaurus clear",
    "serve": "docusaurus serve",
    "write-translations": "docusaurus write-translations",
    "write-heading-ids": "docusaurus write-heading-ids",
    "typecheck": "tsc",
    "lint": "eslint src/ --ext .ts,.tsx,.js,.jsx",
    "lint:fix": "eslint src/ --ext .ts,.tsx,.js,.jsx --fix",
    "format": "prettier --write \"**/*.{js,jsx,ts,tsx,md,mdx,json,yml,yaml}\"",
    "format:check": "prettier --check \"**/*.{js,jsx,ts,tsx,md,mdx,json,yml,yaml}\"",
    "test": "jest",
    "test:coverage": "jest --coverage",
    "security:audit": "npm audit --audit-level high",
    "production:build": "cross-env NODE_ENV=production npm run build",
    "production:validate": "npm run typecheck && npm run lint && npm run test:coverage",
    "production:deploy": "npm run production:validate && npm run production:build && npm run deploy"
  },
  "dependencies": {
    "@docusaurus/core": "^3.0.0",
    "@docusaurus/preset-classic": "^3.0.0",
    "@docusaurus/module-type-aliases": "^3.0.0",
    "@docusaurus/types": "^3.0.0",
    "@docusaurus/plugin-sitemap": "^3.0.0",
    "@docusaurus/plugin-google-analytics": "^3.0.0",
    "@docusaurus/plugin-pwa": "^3.0.0",
    "clsx": "^2.0.0",
    "prism-react-renderer": "^2.3.0",
    "react": "^18.0.0",
    "react-dom": "^18.0.0",
    "web-vitals": "^3.5.0"
  },
  "devDependencies": {
    "@docusaurus/eslint-plugin": "^3.0.0",
    "@docusaurus/tsconfig": "^3.0.0",
    "@docusaurus/types": "^3.0.0",
    "@types/react": "^18.0.0",
    "@types/node": "^18.0.0",
    "cross-env": "^7.0.3",
    "eslint": "^8.0.0",
    "eslint-config-custom": "workspace:*",
    "prettier": "^3.0.0",
    "typescript": "~5.2.0",
    "jest": "^29.0.0",
    "@types/jest": "^29.0.0",
    "ts-jest": "^29.0.0",
    "jest-environment-jsdom": "^29.0.0",
    "jest-axe": "^8.0.0",
    "@axe-core/react": "^4.10.0"
  },
  "browserslist": {
    "production": [
      ">0.5%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 3 chrome version",
      "last 3 firefox version",
      "last 3 safari version"
    ]
  },
  "engines": {
    "node": ">=18.0"
  }
}
```

### Build Optimization

Optimize the build for production:

```typescript
// docusaurus.config.ts (production-specific optimizations)
import { Config } from '@docusaurus/types';

const config: Config = {
  // ... previous configuration

  plugins: [
    // ... previous plugins

    // Plugin for bundle analysis (useful for optimization)
    [
      '@docusaurus/plugin-client-redirects',
      {
        createRedirects(existingPath) {
          // Create redirects for SEO and user experience
          if (existingPath.includes('/old-section/')) {
            return [existingPath.replace('/old-section/', '/new-section/')];
          }
          return undefined;
        },
      },
    ],

    // Plugin for minification and optimization
    [
      '@docusaurus/plugin-content-docs',
      {
        path: 'docs',
        routeBasePath: '/',
        sidebarPath: './sidebars.ts',
        editUrl: 'https://github.com/your-org/physical-ai-textbook/edit/main/',
        showLastUpdateTime: true,
        showLastUpdateAuthor: true,
        // Enable search indexing
        searchMetadata: {
          version: '1.0.0',
          tag: 'textbook',
        },
      },
    ],
  ],

  themes: [
    [
      '@docusaurus/theme-search-algolia',
      {
        // Algolia search configuration
        appId: 'YOUR_ALGOLIA_APP_ID',
        apiKey: 'YOUR_ALGOLIA_SEARCH_KEY',
        indexName: 'physical-ai-textbook',
        contextualSearch: true,
        searchParameters: {},
        searchPagePath: 'search',
      },
    ],
  ],
};

export default config;
```

## Deployment Strategies

### GitHub Pages Deployment

Configure GitHub Pages deployment:

```yaml
# .github/workflows/deploy.yml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'
          cache: 'npm'

      - name: Setup Pages
        uses: actions/configure-pages@v4

      - name: Install dependencies
        run: |
          cd docs
          npm ci

      - name: Build with Docusaurus
        run: |
          cd docs
          npm run build

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: docs/build

  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

### Netlify Deployment Configuration

Create Netlify configuration:

```toml
# netlify.toml
[build]
  base = "docs"
  publish = "build"
  command = "npm run build"

[build.environment]
  NODE_VERSION = "18"
  NPM_FLAGS = "--version" # Enable NPM version output

[[redirects]]
  from = "/docs/*"
  to = "/:splat"
  status = 200

[[headers]]
  for = "/*"
  [headers.values]
    X-Frame-Options = "DENY"
    X-Content-Type-Options = "nosniff"
    Referrer-Policy = "strict-origin-when-cross-origin"
    Permissions-Policy = "geolocation=(), microphone=(), camera=()"
    Content-Security-Policy = "default-src 'self'; script-src 'self' 'unsafe-inline' https://www.google-analytics.com; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' https://fonts.gstatic.com; connect-src 'self' https://www.google-analytics.com;"
    Strict-Transport-Security = "max-age=63072000; includeSubDomains; preload"

[context.deploy-preview]
  command = "npm run build -- --draft"

[context.branch-deploy]
  command = "npm run build -- --draft"
```

### Vercel Deployment Configuration

Create Vercel configuration:

```json
// docs/vercel.json
{
  "version": 2,
  "framework": "docusaurus",
  "functions": {
    "src/pages/api/**/*.js": {
      "memory": 1024,
      "maxDuration": 10
    }
  },
  "routes": [
    {
      "src": "/docs/(.*)",
      "dest": "/$1",
      "headers": {
        "X-Frame-Options": "DENY",
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline' https://www.google-analytics.com; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' https://fonts.gstatic.com; connect-src 'self' https://www.google-analytics.com;"
      }
    }
  ],
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        {
          "key": "X-Frame-Options",
          "value": "DENY"
        },
        {
          "key": "X-Content-Type-Options",
          "value": "nosniff"
        },
        {
          "key": "Referrer-Policy",
          "value": "strict-origin-when-cross-origin"
        }
      ]
    }
  ]
}
```

## Security Configuration

### Content Security Policy

Implement comprehensive security headers:

```typescript
// src/theme/SecurityHeaders.tsx
import React from 'react';

const SecurityHeaders: React.FC = () => {
  React.useEffect(() => {
    // This would be implemented differently in a real scenario
    // as Docusaurus generates static HTML
  }, []);

  return null;
};

export default SecurityHeaders;
```

### Security Headers Configuration

Configure security headers for production:

```javascript
// For Express.js server (if needed for proxy or API)
const helmet = require('helmet');

app.use(
  helmet({
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ["'self'"],
        styleSrc: ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
        fontSrc: ["'self'", "https://fonts.gstatic.com"],
        imgSrc: ["'self'", "data:", "https:"],
        scriptSrc: ["'self'", "'unsafe-inline'", "https://www.google-analytics.com"],
        connectSrc: ["'self'", "https://www.google-analytics.com"],
        frameSrc: ["'none'"],
      },
    },
    hsts: {
      maxAge: 63072000,
      includeSubDomains: true,
      preload: true,
    },
    referrerPolicy: {
      policy: 'strict-origin-when-cross-origin',
    },
    frameguard: {
      action: 'deny',
    },
  })
);
```

### Authentication and Authorization

For protected content areas:

```typescript
// src/utils/auth.ts
export interface UserSession {
  id: string;
  email: string;
  role: 'student' | 'instructor' | 'admin';
  permissions: string[];
  expiresAt: Date;
}

export class AuthService {
  private static readonly TOKEN_KEY = 'textbook_auth_token';
  private static readonly USER_KEY = 'textbook_user_data';

  static isAuthenticated(): boolean {
    const token = localStorage.getItem(this.TOKEN_KEY);
    const userData = localStorage.getItem(this.USER_KEY);

    if (!token || !userData) {
      return false;
    }

    try {
      const user: UserSession = JSON.parse(userData);
      return new Date() < user.expiresAt;
    } catch {
      this.clearAuth();
      return false;
    }
  }

  static getUser(): UserSession | null {
    if (!this.isAuthenticated()) {
      return null;
    }

    try {
      const userData = localStorage.getItem(this.USER_KEY);
      return userData ? JSON.parse(userData) : null;
    } catch {
      return null;
    }
  }

  static setAuth(token: string, user: UserSession): void {
    localStorage.setItem(this.TOKEN_KEY, token);
    localStorage.setItem(this.USER_KEY, JSON.stringify(user));
  }

  static clearAuth(): void {
    localStorage.removeItem(this.TOKEN_KEY);
    localStorage.removeItem(this.USER_KEY);
  }

  static hasPermission(permission: string): boolean {
    const user = this.getUser();
    return user?.permissions.includes(permission) ?? false;
  }

  static hasRole(role: string): boolean {
    const user = this.getUser();
    return user?.role === role;
  }
}
```

## Performance Optimization

### Asset Optimization

Optimize assets for production:

```typescript
// src/utils/assetOptimizer.ts
export class AssetOptimizer {
  static optimizeImage(src: string, maxWidth: number = 1920): string {
    // For static hosting, this would be handled at build time
    // This is a placeholder for CDN-based optimization
    if (process.env.NODE_ENV === 'production') {
      // Return optimized version if available
      const optimizedSrc = src.replace(/\.(jpg|jpeg|png|webp)$/i, `_${maxWidth}w.$1`);
      return optimizedSrc;
    }
    return src;
  }

  static preloadCriticalAssets(): void {
    // Preload critical assets for better performance
    const criticalAssets = [
      '/css/main.css',
      '/js/main.js',
      '/fonts/main-font.woff2'
    ];

    criticalAssets.forEach(asset => {
      const link = document.createElement('link');
      link.rel = 'preload';
      link.href = asset;

      if (asset.endsWith('.css')) {
        link.as = 'style';
      } else if (asset.endsWith('.js')) {
        link.as = 'script';
      } else if (asset.endsWith('.woff2')) {
        link.as = 'font';
        link.type = 'font/woff2';
        link.crossOrigin = 'anonymous';
      }

      document.head.appendChild(link);
    });
  }

  static lazyLoadNonCriticalImages(): void {
    // Use Intersection Observer for lazy loading
    if ('IntersectionObserver' in window) {
      const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const img = entry.target as HTMLImageElement;
            if (img.dataset.src) {
              img.src = img.dataset.src;
              img.removeAttribute('data-src');
            }
            if (img.dataset.srcset) {
              img.srcset = img.dataset.srcset;
              img.removeAttribute('data-srcset');
            }
            observer.unobserve(img);
          }
        });
      });

      document.querySelectorAll('img[data-src]').forEach(img => {
        imageObserver.observe(img);
      });
    }
  }
}
```

### Caching Strategy

Implement caching strategies:

```typescript
// src/service-worker.ts
/// <reference lib="webworker" />
declare const self: ServiceWorkerGlobalScope;

const CACHE_NAME = 'textbook-v1';
const STATIC_ASSETS = [
  '/',
  '/css/main.css',
  '/js/main.js',
  '/offline.html',
  '/manifest.json'
];

self.addEventListener('install', (event: ExtendableEvent) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(STATIC_ASSETS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('fetch', (event: FetchEvent) => {
  const { request } = event;

  // Don't cache admin routes or API calls
  if (request.url.includes('/api/') || request.url.includes('/admin/')) {
    return;
  }

  event.respondWith(
    caches.match(request)
      .then((response) => {
        // Return cached version if available
        if (response) {
          return response;
        }

        // Clone the request for network fetch
        const fetchRequest = request.clone();

        return fetch(fetchRequest)
          .then((networkResponse) => {
            // Don't cache error responses
            if (!networkResponse || networkResponse.status !== 200) {
              return networkResponse;
            }

            // Clone the response for caching
            const responseToCache = networkResponse.clone();

            caches.open(CACHE_NAME)
              .then((cache) => {
                cache.put(request, responseToCache);
              });

            return networkResponse;
          })
          .catch(() => {
            // Return offline page for navigation requests
            if (request.mode === 'navigate') {
              return caches.match('/offline.html');
            }

            // Return error for other requests
            return new Response('Network Error', {
              status: 503,
              statusText: 'Service Unavailable',
              headers: new Headers({ 'Content-Type': 'text/html' })
            });
          });
      })
  );
});

self.addEventListener('activate', (event: ExtendableEvent) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});
```

## Monitoring and Analytics

### Performance Monitoring

Set up comprehensive performance monitoring:

```typescript
// src/utils/monitoring.ts
import { getCLS, getFID, getFCP, getLCP, getTTFB } from 'web-vitals';

export interface PerformanceMetrics {
  cls: number;
  fid: number;
  fcp: number;
  lcp: number;
  ttfb: number;
  navigationTiming: PerformanceNavigationTiming;
}

export class PerformanceMonitoring {
  private static readonly ANALYTICS_ENDPOINT = '/api/analytics/performance';
  private static readonly SAMPLE_RATE = 0.1; // Sample 10% of users

  static initialize(): void {
    // Only initialize for production
    if (process.env.NODE_ENV !== 'production') {
      return;
    }

    // Track core web vitals
    this.trackCoreWebVitals();

    // Track custom metrics
    this.trackCustomMetrics();

    // Set up error tracking
    this.setupErrorTracking();
  }

  private static trackCoreWebVitals(): void {
    getCLS(this.sendMetric.bind(this));
    getFID(this.sendMetric.bind(this));
    getFCP(this.sendMetric.bind(this));
    getLCP(this.sendMetric.bind(this));
    getTTFB(this.sendMetric.bind(this));
  }

  private static sendMetric(metric: any): void {
    // Sample the metric
    if (Math.random() > this.SAMPLE_RATE) {
      return;
    }

    // Send to analytics endpoint
    const payload = {
      metric: metric.name,
      value: metric.value,
      id: metric.id,
      timestamp: Date.now(),
      userAgent: navigator.userAgent,
      url: window.location.href,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight
      }
    };

    // Use sendBeacon for better reliability
    if (navigator.sendBeacon) {
      navigator.sendBeacon(
        this.ANALYTICS_ENDPOINT,
        JSON.stringify(payload)
      );
    } else {
      // Fallback to fetch
      fetch(this.ANALYTICS_ENDPOINT, {
        method: 'POST',
        body: JSON.stringify(payload),
        headers: { 'Content-Type': 'application/json' },
        keepalive: true // Keep connection alive for slow networks
      }).catch(console.error);
    }
  }

  private static trackCustomMetrics(): void {
    // Track time to interactive
    const trackTimeToInteractive = () => {
      if (document.readyState === 'complete') {
        const tti = performance.now();
        this.sendMetric({
          name: 'tti',
          value: tti,
          id: 'time-to-interactive'
        });
      } else {
        window.addEventListener('load', trackTimeToInteractive);
      }
    };
    trackTimeToInteractive();

    // Track resource loading times
    window.addEventListener('load', () => {
      const resources = performance.getEntriesByType('navigation');
      resources.forEach((resource: any) => {
        this.sendMetric({
          name: 'resource_load_time',
          value: resource.loadEventEnd - resource.fetchStart,
          id: 'resource_load'
        });
      });
    });
  }

  private static setupErrorTracking(): void {
    // Track unhandled errors
    window.addEventListener('error', (event) => {
      this.sendError({
        type: 'javascript_error',
        message: event.message,
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        stack: event.error?.stack,
        url: window.location.href,
        userAgent: navigator.userAgent
      });
    });

    // Track unhandled promise rejections
    window.addEventListener('unhandledrejection', (event) => {
      this.sendError({
        type: 'promise_rejection',
        message: event.reason?.toString?.() || 'Unhandled Promise Rejection',
        stack: event.reason?.stack,
        url: window.location.href
      });
    });
  }

  private static sendError(error: any): void {
    const payload = {
      ...error,
      timestamp: Date.now(),
      userAgent: navigator.userAgent,
      url: window.location.href
    };

    if (navigator.sendBeacon) {
      navigator.sendBeacon('/api/analytics/errors', JSON.stringify(payload));
    } else {
      fetch('/api/analytics/errors', {
        method: 'POST',
        body: JSON.stringify(payload),
        headers: { 'Content-Type': 'application/json' },
        keepalive: true
      }).catch(console.error);
    }
  }

  static trackUserInteraction(element: string, action: string): void {
    if (Math.random() > this.SAMPLE_RATE) {
      return;
    }

    const payload = {
      type: 'user_interaction',
      element,
      action,
      timestamp: Date.now(),
      url: window.location.href,
      userAgent: navigator.userAgent
    };

    if (navigator.sendBeacon) {
      navigator.sendBeacon('/api/analytics/interactions', JSON.stringify(payload));
    }
  }
}
```

### Analytics Implementation

Implement analytics for educational metrics:

```typescript
// src/utils/analytics.ts
export interface LearningEvent {
  type: 'page_view' | 'lesson_start' | 'lesson_complete' | 'assessment_submit' | 'video_play' | 'resource_download';
  userId?: string;
  lessonId?: string;
  courseId?: string;
  timestamp: number;
  metadata: Record<string, any>;
}

export class LearningAnalytics {
  private static readonly TRACKING_ENDPOINT = '/api/analytics/learning';
  private static readonly BATCH_SIZE = 10;
  private static readonly BATCH_INTERVAL = 5000; // 5 seconds

  private static eventQueue: LearningEvent[] = [];
  private static batchTimer: NodeJS.Timeout | null = null;

  static initialize(): void {
    // Initialize analytics tracking
    if (process.env.NODE_ENV !== 'production') {
      return;
    }

    // Set up batching
    this.batchTimer = setInterval(() => {
      this.flushEvents();
    }, this.BATCH_INTERVAL);

    // Track page views
    this.trackPageView();

    // Track user interactions
    this.setupInteractionTracking();
  }

  static track(type: LearningEvent['type'], metadata: Record<string, any> = {}): void {
    const event: LearningEvent = {
      type,
      timestamp: Date.now(),
      metadata,
      ...this.getContext()
    };

    this.eventQueue.push(event);

    // Flush if batch is full
    if (this.eventQueue.length >= this.BATCH_SIZE) {
      this.flushEvents();
    }
  }

  static trackLessonStart(lessonId: string, courseId: string): void {
    this.track('lesson_start', { lessonId, courseId });
  }

  static trackLessonComplete(lessonId: string, courseId: string, duration: number, score?: number): void {
    this.track('lesson_complete', {
      lessonId,
      courseId,
      duration,
      score: score || null
    });
  }

  static trackAssessmentSubmit(lessonId: string, answers: number[], score: number, timeTaken: number): void {
    this.track('assessment_submit', {
      lessonId,
      answers,
      score,
      timeTaken
    });
  }

  private static getContext(): Partial<LearningEvent> {
    return {
      userId: this.getUserId(),
      url: window.location.href,
      userAgent: navigator.userAgent,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight
      },
      referrer: document.referrer
    };
  }

  private static getUserId(): string | undefined {
    // Get user ID from auth system or anonymous ID
    return localStorage.getItem('user_id') || this.generateAnonymousId();
  }

  private static generateAnonymousId(): string {
    if (!localStorage.getItem('anonymous_id')) {
      const id = 'anon_' + Date.now().toString(36) + Math.random().toString(36).substr(2);
      localStorage.setItem('anonymous_id', id);
    }
    return localStorage.getItem('anonymous_id')!;
  }

  private static flushEvents(): void {
    if (this.eventQueue.length === 0) {
      return;
    }

    const events = [...this.eventQueue];
    this.eventQueue = [];

    fetch(this.TRACKING_ENDPOINT, {
      method: 'POST',
      body: JSON.stringify({ events }),
      headers: { 'Content-Type': 'application/json' },
      keepalive: true
    }).catch(err => {
      console.error('Failed to send analytics:', err);
      // Add events back to queue for retry
      this.eventQueue = [...events, ...this.eventQueue];
    });
  }

  private static trackPageView(): void {
    // Track initial page view
    this.track('page_view', {
      path: window.location.pathname,
      referrer: document.referrer
    });

    // Track navigation changes
    const originalPushState = history.pushState;
    history.pushState = function(...args) {
      originalPushState.apply(history, args);
      setTimeout(() => {
        LearningAnalytics.track('page_view', {
          path: window.location.pathname,
          referrer: document.referrer
        });
      }, 0);
    };
  }

  private static setupInteractionTracking(): void {
    // Track clicks on interactive elements
    document.addEventListener('click', (event) => {
      const target = event.target as HTMLElement;

      // Track clicks on assessment options
      if (target.closest('[data-testid="assessment-option"]')) {
        const assessment = target.closest('[data-testid="assessment-question"]');
        if (assessment) {
          const questionId = assessment.getAttribute('data-question-id');
          this.track('assessment_interaction', {
            questionId,
            action: 'click_option'
          });
        }
      }

      // Track clicks on interactive demos
      if (target.closest('[data-testid="interactive-demo"]')) {
        this.track('interactive_demo_interaction', {
          action: 'click'
        });
      }
    });
  }

  static shutdown(): void {
    // Flush remaining events
    if (this.batchTimer) {
      clearInterval(this.batchTimer);
    }
    this.flushEvents();
  }
}
```

## Disaster Recovery and Rollback

### Backup Procedures

```bash
# backup-production.sh
#!/bin/bash

# Backup production configuration and content
BACKUP_DIR="/backups/textbook-$(date +%Y%m%d-%H%M%S)"
mkdir -p $BACKUP_DIR

# Backup GitHub repository
git clone https://github.com/your-org/physical-ai-textbook.git $BACKUP_DIR/source

# Backup deployment configuration
cp -r .github $BACKUP_DIR/config/
cp -r docs $BACKUP_DIR/docs/
cp netlify.toml $BACKUP_DIR/config/ 2>/dev/null || echo "No netlify.toml found"
cp vercel.json $BACKUP_DIR/config/ 2>/dev/null || echo "No vercel.json found"

# Backup analytics data (if stored separately)
# This would depend on your specific analytics setup

echo "Backup completed: $BACKUP_DIR"
```

### Rollback Procedures

```bash
# rollback-production.sh
#!/bin/bash

# Rollback to a previous version
if [ $# -eq 0 ]; then
    echo "Usage: $0 <backup-directory>"
    exit 1
fi

BACKUP_DIR=$1

if [ ! -d "$BACKUP_DIR" ]; then
    echo "Backup directory does not exist: $BACKUP_DIR"
    exit 1
fi

echo "Rolling back from: $BACKUP_DIR"

# For GitHub Pages deployment, rollback means reverting git commits
cd $BACKUP_DIR/source

# Get the last good commit from the backup metadata
LAST_GOOD_COMMIT=$(cat $BACKUP_DIR/metadata/last_good_commit.txt 2>/dev/null || echo "HEAD~1")

echo "Reverting to commit: $LAST_GOOD_COMMIT"
git reset --hard $LAST_GOOD_COMMIT

# Force push to main branch (be careful!)
git push origin main --force

echo "Rollback completed. Deployment should trigger automatically."
```

### Health Check Scripts

```typescript
// scripts/health-check.ts
import axios from 'axios';
import { performance } from 'perf_hooks';

interface HealthCheckResult {
  url: string;
  status: number;
  responseTime: number;
  errors: string[];
}

async function checkHealth(url: string): Promise<HealthCheckResult> {
  const startTime = performance.now();
  const errors: string[] = [];

  try {
    const response = await axios.get(url, {
      timeout: 10000, // 10 second timeout
      validateStatus: (status) => status < 500 // Accept 4xx as "healthy" for some checks
    });

    const responseTime = performance.now() - startTime;

    // Check for common issues
    if (response.data.includes('Error')) {
      errors.push('Page contains error text');
    }

    if (response.data.toLowerCase().includes('maintenance')) {
      errors.push('Page indicates maintenance mode');
    }

    if (responseTime > 5000) {
      errors.push(`Slow response time: ${Math.round(responseTime)}ms`);
    }

    return {
      url,
      status: response.status,
      responseTime: Math.round(responseTime),
      errors
    };
  } catch (error: any) {
    const responseTime = performance.now() - startTime;
    errors.push(error.message || 'Unknown error');

    return {
      url,
      status: error.response?.status || 0,
      responseTime: Math.round(responseTime),
      errors
    };
  }
}

async function runHealthChecks(): Promise<void> {
  const urlsToCheck = [
    'https://your-domain.com/',
    'https://your-domain.com/docs/week-01-02-introduction/foundations-of-physical-ai',
    'https://your-domain.com/docs/week-03-05-ros2/ros2-architecture-concepts',
    'https://your-domain.com/search',
    'https://your-domain.com/sitemap.xml'
  ];

  console.log('Running health checks...\n');

  const results = await Promise.all(
    urlsToCheck.map(url => checkHealth(url))
  );

  let healthyCount = 0;
  results.forEach(result => {
    const status = result.errors.length === 0 ? '✅ HEALTHY' : '❌ UNHEALTHY';
    console.log(`${status} ${result.url}`);
    console.log(`  Status: ${result.status}, Response Time: ${result.responseTime}ms`);

    if (result.errors.length > 0) {
      console.log(`  Errors: ${result.errors.join(', ')}`);
    }
    console.log('');

    if (result.errors.length === 0) {
      healthyCount++;
    }
  });

  console.log(`Health check summary: ${healthyCount}/${urlsToCheck.length} URLs healthy`);

  if (healthyCount < urlsToCheck.length) {
    process.exit(1); // Fail the check if any URLs are unhealthy
  }
}

// Run health check
runHealthChecks().catch(error => {
  console.error('Health check failed:', error);
  process.exit(1);
});
```

## Maintenance Procedures

### Routine Maintenance

Create a maintenance checklist:

```markdown
# Production Maintenance Checklist

## Daily Checks
- [ ] Monitor uptime with external service
- [ ] Check analytics for unusual patterns
- [ ] Review error logs
- [ ] Verify search functionality
- [ ] Test key user journeys

## Weekly Checks
- [ ] Review performance metrics
- [ ] Check for broken links
- [ ] Update content based on user feedback
- [ ] Review and clean up analytics data
- [ ] Update dependencies (security patches)

## Monthly Checks
- [ ] Full backup verification
- [ ] Performance audit and optimization
- [ ] Security scan and updates
- [ ] User feedback analysis
- [ ] Content quality review
- [ ] Accessibility audit

## Quarterly Reviews
- [ ] Architecture review and optimization
- [ ] Infrastructure scaling assessment
- [ ] Feature usage analysis
- [ ] Learning outcome evaluation
- [ ] Technology stack update planning
- [ ] Security penetration testing
```

## Summary

Production deployment and maintenance of the Physical AI & Humanoid Robotics textbook involves:

1. **Architecture**: Static site hosting with CDN distribution
2. **Security**: Content Security Policy, authentication, secure headers
3. **Performance**: Asset optimization, caching, lazy loading
4. **Monitoring**: Performance metrics, error tracking, user analytics
5. **Disaster Recovery**: Backup procedures, rollback plans, health checks
6. **Maintenance**: Regular checks, updates, and optimization
7. **Analytics**: Learning metrics, user engagement, content effectiveness

By following these production configuration guidelines, the Physical AI & Humanoid Robotics textbook will be deployed securely, performantly, and maintainably, providing an excellent learning experience for all users.