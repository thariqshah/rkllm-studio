FROM node:20-alpine

WORKDIR /app

# Install dependencies first
COPY package*.json ./
RUN npm install

# Copy source
COPY . .

EXPOSE 3000

# Run Vite with host binding
CMD ["npm", "run", "dev", "--", "--host"]
