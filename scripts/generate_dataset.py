"""
Generate 1000 training examples for ad generation
Uses variations and augmentation to create realistic dataset
"""
import pandas as pd
import random

# Base templates for different ad styles
AD_STYLES = {
    "benefits_focused": [
        "{name} delivers {benefit1} and {benefit2}. {description}. Perfect for {audience}. {cta}!",
        "Experience {benefit1} with {name}. {description}. {benefit2} included. {cta} today!",
        "Get {benefit1} and {benefit2} with our {name}. {description}. {cta} now!",
    ],
    "problem_solution": [
        "Tired of {problem}? {name} is your solution! {description}. {benefit}. {cta}!",
        "Say goodbye to {problem}. {name} {description}. {benefit}. {cta} today!",
        "Struggling with {problem}? Our {name} changes everything. {description}. {cta}!",
    ],
    "social_proof": [
        "Join {number}+ happy customers! {name} - {description}. {benefit}. {cta}!",
        "{name}: Rated 5 stars by {number}+ users. {description}. {benefit}. {cta} now!",
        "Trusted by {number}+ customers. {name} {description}. {cta} today!",
    ],
    "urgency": [
        "Limited time! Get {name} now. {description}. {benefit}. {cta} before it's gone!",
        "Don't miss out on {name}. {description}. {benefit}. {cta} - stock is limited!",
        "Last chance! {name} {description}. {benefit}. {cta} while supplies last!",
    ],
    "feature_rich": [
        "{name} features {feature1}, {feature2}, and {feature3}. {description}. {cta}!",
        "Packed with {feature1}, {feature2}, and more. {name} {description}. {cta} today!",
        "{name} includes {feature1} and {feature2}. {description}. {cta} now!",
    ]
}

# Word banks for variation
BENEFITS = [
    "save time", "save money", "improve quality", "boost productivity", "enhance performance",
    "increase comfort", "reduce stress", "simplify life", "eliminate hassle", "maximize results"
]

PROBLEMS = [
    "wasted time", "high costs", "poor quality", "complicated setups", "uncomfortable use",
    "frequent breakdowns", "limited features", "outdated technology", "unreliable performance"
]

AUDIENCES = [
    "professionals", "families", "students", "athletes", "home chefs", "tech enthusiasts",
    "busy parents", "fitness lovers", "gamers", "travelers", "remote workers"
]

CTAS = [
    "Order now", "Buy today", "Shop now", "Get yours", "Try it", "Upgrade now",
    "Start saving", "Experience the difference", "Make the switch", "Don't wait"
]

NUMBERS = ["10,000", "25,000", "50,000", "100,000", "150,000"]

# 100 product templates
PRODUCTS = [
    # Electronics (20)
    ("Wireless Headphones", "Electronics", "Premium sound with noise cancellation and long battery"),
    ("Smart Watch", "Wearables", "Fitness tracking with heart rate monitor and GPS"),
    ("Bluetooth Speaker", "Electronics", "Portable waterproof speaker with 360-degree sound"),
    ("Wireless Earbuds", "Electronics", "True wireless with charging case and touch controls"),
    ("Power Bank", "Electronics", "High-capacity portable charger with fast charging"),
    ("Gaming Mouse", "Electronics", "Precision wireless mouse with RGB lighting"),
    ("Mechanical Keyboard", "Electronics", "Tactile switches with customizable backlighting"),
    ("Webcam HD", "Electronics", "1080p camera with auto-focus and built-in microphone"),
    ("USB-C Hub", "Electronics", "Multi-port adapter with HDMI and USB 3.0"),
    ("Wireless Charger", "Electronics", "Fast charging pad for all Qi-enabled devices"),
    ("Smart Light Bulbs", "Electronics", "WiFi-enabled color-changing LED bulbs"),
    ("Ring Light", "Electronics", "Adjustable brightness for perfect lighting"),
    ("Phone Gimbal", "Electronics", "3-axis stabilizer for smooth video recording"),
    ("Dash Cam", "Electronics", "HD car camera with loop recording and night vision"),
    ("Tablet Stand", "Electronics", "Adjustable aluminum stand for all devices"),
    ("Cable Organizer", "Electronics", "Keep all cables tangle-free and organized"),
    ("Screen Protector", "Electronics", "Tempered glass protection for phones"),
    ("Phone Case", "Electronics", "Durable protection with wireless charging support"),
    ("Laptop Cooling Pad", "Electronics", "Dual fans to prevent overheating"),
    ("Smart Plug", "Electronics", "WiFi-enabled outlet with voice control"),
    
    # Home & Kitchen (20)
    ("Stainless Steel Water Bottle", "Home & Kitchen", "Insulated to keep drinks cold or hot all day"),
    ("Coffee Maker", "Home & Kitchen", "Programmable machine that brews perfect coffee"),
    ("Air Fryer", "Home & Kitchen", "Healthier cooking with little to no oil needed"),
    ("Blender", "Home & Kitchen", "Powerful motor for smooth smoothies and shakes"),
    ("Knife Set", "Home & Kitchen", "Professional-grade stainless steel blades"),
    ("Cutting Board Set", "Home & Kitchen", "Bamboo boards in multiple sizes"),
    ("Measuring Cups", "Home & Kitchen", "Stainless steel for accurate measurements"),
    ("Mixing Bowls", "Home & Kitchen", "Nesting set with non-slip bases"),
    ("Kitchen Scale", "Home & Kitchen", "Digital precision for perfect recipes"),
    ("Food Storage Containers", "Home & Kitchen", "Air-tight BPA-free containers"),
    ("Instant Pot", "Home & Kitchen", "7-in-1 pressure cooker for fast meals"),
    ("Electric Kettle", "Home & Kitchen", "Fast boiling with auto shut-off safety"),
    ("Toaster Oven", "Home & Kitchen", "Compact convection oven for any task"),
    ("Slow Cooker", "Home & Kitchen", "Set-and-forget cooking for busy families"),
    ("French Press", "Home & Kitchen", "Brew rich coffee in minutes"),
    ("Tea Infuser", "Home & Kitchen", "Stainless mesh for loose leaf tea"),
    ("Spice Rack", "Home & Kitchen", "Organize all spices with rotating stand"),
    ("Wine Opener", "Home & Kitchen", "Electric corkscrew for effortless opening"),
    ("Can Opener", "Home & Kitchen", "Smooth-edge safe opener"),
    ("Trash Bags", "Home & Kitchen", "Heavy-duty leak-proof bags"),
    
    # Sports & Fitness (15)
    ("Yoga Mat", "Sports & Fitness", "Non-slip cushioned mat for all exercises"),
    ("Resistance Bands", "Sports & Fitness", "Complete set with multiple resistance levels"),
    ("Dumb bells", "Sports & Fitness", "Adjustable weights for home workouts"),
    ("Jump Rope", "Sports & Fitness", "Speed rope for cardio training"),
    ("Foam Roller", "Sports & Fitness", "Muscle recovery and massage tool"),
    ("Yoga Blocks", "Sports & Fitness", "Supportive props for proper alignment"),
    ("Exercise Ball", "Sports & Fitness", "Stability ball for core workouts"),
    ("Kettlebell", "Sports & Fitness", "Cast iron weight for strength training"),
    ("Pull-Up Bar", "Sports & Fitness", "Doorway-mounted bar for upper body"),
    ("Ab Wheel", "Sports & Fitness", "Core strengthening roller"),
    ("Boxing Gloves", "Sports & Fitness", "Padded gloves for training"),
    ("Ankle Weights", "Sports & Fitness", "Adjustable weights for leg exercises"),
    ("Yoga Strap", "Sports & Fitness", "Improve flexibility and reach"),
    ("Balance Board", "Sports & Fitness", "Wooden board for stability training"),
    ("Push-Up Bars", "Sports & Fitness", "Elevated handles for better form"),
    
    # Personal Care (15)
    ("Electric Toothbrush", "Personal Care", "Sonic cleaning with smart timer"),
    ("Hair Dryer", "Personal Care", "Ionic technology for shinier hair"),
    ("Facial Cleanser", "Personal Care", "Gentle formula for all skin types"),
    ("Moisturizer", "Personal Care", "Hydrating cream with SPF protection"),
    ("Sunscreen SPF 50", "Personal Care", "Broad-spectrum UV protection"),
    ("Lip Balm Set", "Personal Care", "Moisturizing balms in multiple flavors"),
    ("Body Lotion", "Personal Care", "Nourishing formula for soft skin"),
    ("Shampoo", "Personal Care", "Sulfate-free for healthy hair"),
    ("Conditioner", "Personal Care", "Deep conditioning treatment"),
    ("Face Masks", "Personal Care", "Hydrating sheet masks for spa days"),
    ("Makeup Brushes", "Personal Care", "Professional quality synthetic bristles"),
    ("Nail Clippers", "Personal Care", "Precision stainless steel set"),
    ("Tweezers", "Personal Care", "Slant-tip for perfect plucking"),
    ("Razor", "Personal Care", "Multi-blade for smooth shave"),
    ("Cologne", "Personal Care", "Long-lasting sophisticated scent"),
    
    # Clothing (10)
    ("T-Shirt", "Clothing", "100% organic cotton in multiple colors"),
    ("Hoodie", "Clothing", "Warm fleece with kangaroo pocket"),
    ("Jeans", "Clothing", "Classic fit denim that lasts"),
    ("Socks Pack", "Clothing", "Comfortable cushioned athletic socks"),
    ("Running Shorts", "Clothing", "Moisture-wicking with pockets"),
    ("Tank Top", "Clothing", "Breathable fabric for workouts"),
    ("Sweatpants", "Clothing", "Cozy joggers for lounging"),
    ("Jacket", "Clothing", "Waterproof windbreaker for outdoors"),
    ("Hat", "Clothing", "Adjustable baseball cap"),
    ("Gloves", "Clothing", "Touchscreen-compatible winter gloves"),
    
    # Accessories (10)
    ("Backpack", "Accessories", "Waterproof with laptop compartment"),
    ("Wallet", "Accessories", "Slim bifold genuine leather"),
    ("Sunglasses", "Accessories", "Polarized UV400 protection"),
    ("Watch", "Accessories", "Classic analog with leather strap"),
    ("Belt", "Accessories", "Reversible genuine leather belt"),
    ("Umbrella", "Accessories", "Compact auto-open windproof"),
    ("Keychain", "Accessories", "Multi-tool with LED light"),
    ("Tote Bag", "Accessories", "Canvas reusable shopping bag"),
    ("Scarf", "Accessories", "Soft cashmere blend"),
    ("Tie", "Accessories", "Silk necktie in classic patterns"),
    
    # Office Supplies (10)
    ("Desk Lamp", "Office Supplies", "LED with adjustable brightness"),
    ("Notebook", "Office Supplies", "Hardcover journal with lined pages"),
    ("Pens", "Office Supplies", "Smooth-writing ballpoint pens"),
    ("Highlighters", "Office Supplies", "Vibrant colors that don't bleed"),
    ("Sticky Notes", "Office Supplies", "Adhesive notes in assorted sizes"),
    ("Desk Organizer", "Office Supplies", "Bamboo organizer with compartments"),
    ("Mouse Pad", "Office Supplies", "Large gaming surface"),
    ("Stapler", "Office Supplies", "Heavy-duty metal construction"),
    ("Binder Clips", "Office Supplies", "Assorted sizes for any project"),
    ("File Folders", "Office Supplies", "Durable manila folders"),
]

def generate_dataset(num_samples=1000):
    """Generate varied training examples"""
    data = []
    
    for i in range(num_samples):
        # Pick random product
        name, category, description = random.choice(PRODUCTS)
        
        # Pick random ad style
        style = random.choice(list(AD_STYLES.keys()))
        template = random.choice(AD_STYLES[style])
        
        # Fill template
        ad = template.format(
            name=name,
            description=description,
            benefit=random.choice(BENEFITS),
            benefit1=random.choice(BENEFITS),
            benefit2=random.choice(BENEFITS),
            problem=random.choice(PROBLEMS),
            audience=random.choice(AUDIENCES),
            cta=random.choice(CTAS),
            number=random.choice(NUMBERS),
            feature1="premium quality",
            feature2="easy setup",
            feature3="lifetime warranty"
        )
        
        data.append({
            "product_name": name,
            "category": category,
            "description": description,
            "ad_copy": ad
        })
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    print("Generating 1000 training examples...")
    df = generate_dataset(1000)
    df.to_csv("data/products_sample.csv", index=False)
    print(f"✅ Generated {len(df)} examples!")
    print("\nSample:")
    print(df.iloc[0])
